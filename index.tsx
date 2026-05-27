import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { createRoot } from "react-dom/client";
import { GoogleGenAI } from "@google/genai";
import ReactMarkdown from "react-markdown";

// --- Types & Interfaces ---

interface Attachment {
  name: string;
  mimeType: string;
  data: string; // Base64 string
}

interface Message {
  role: "user" | "model";
  content: string;
  thoughts?: string; // For the "Thinking" block
  isError?: boolean;
  isStreaming?: boolean; // New flag to indicate active generation
  attachments?: Attachment[];
  molecules?: MoleculeData[]; // Store parsed molecules associated with this message
}

interface MoleculeData {
  name: string;
  smiles: string;
  description: string;
  properties?: {
    molecularWeight?: string;
    logP?: string;
    hbd?: string;
    hba?: string;
    tpsa?: string;
    rotatableBonds?: string;
    [key: string]: string | undefined;
  };
}

interface ModelConfig {
    id: string;
    name: string;
}

// Global declaration for external libraries loaded via CDN
declare global {
  interface Window {
    SmilesDrawer: any;
    $3Dmol: any;
  }
}

// --- Constants ---

const MODELS: ModelConfig[] = [
    // Gemini 3.0 Series (Preview)
    { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash" },
    { id: "gemini-3-flash-preview", name: "Gemini 3.0 Flash" },
    
    // Gemini 2.5 Series (Experimental/Preview)
    { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro" },
    { id: "gemini-3-pro-preview", name: "Gemini 3.0 Pro" },
];

const SYSTEM_INSTRUCTION = `
You are Open Drug, an elite AI Drug Discovery Agent. 

**LANGUAGE ADAPTATION:**
You MUST respond in the same language as the user's latest message. 
- If the user inputs Chinese, your response MUST be in Chinese.
- If the user inputs English, your response MUST be in English.
- If the user switches languages, you MUST switch immediately.
- Default to English ONLY if the user's input language is ambiguous.

**STRICT DOMAIN BOUNDARIES:**
Limit responses to: **Drug Discovery, Medicinal Chemistry, Biology, Pharmacology, and Medical Science**.
Refuse irrelevant topics politely.

**MANDATORY OUTPUT FORMAT:**
Start every response with a "Detailed Thought Process" wrapped in <thinking> tags. 
Include: Deconstruction, Knowledge Retrieval, Search Strategy, Reasoning, and Self-Correction.

Example:
<thinking>
User asking for CDK4 inhibitors...
Checking SAR data...
</thinking>
[Scientific Response]

**JSON Format for Molecules:**
If you discuss specific molecules, append this JSON block strictly at the end:
\`\`\`json
{
  "molecules": [
    {
      "name": "Molecule Name",
      "smiles": "SMILES_STRING",
      "description": "Description",
      "properties": { 
          "molecularWeight": "450.2", 
          "logP": "3.5",
          "hbd": "2",
          "hba": "6",
          "tpsa": "85.4",
          "rotatableBonds": "5"
      }
    }
  ]
}
\`\`\`
`;

// --- Components ---

/**
 * Renders the 2D structure of a molecule using SmilesDrawer.
 */
const Viewer2D = ({ smiles }: { smiles: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current && window.SmilesDrawer) {
      const drawer = new window.SmilesDrawer.Drawer({
        width: canvasRef.current.width,
        height: canvasRef.current.height,
        compactDrawing: false,
        drawingTextColor: '#f8fafc', 
        bondColor: '#94a3b8',
        atomColor: '#38bdf8',
        backgroundColor: 'transparent',
      });
      
      window.SmilesDrawer.parse(smiles, (tree: any) => {
        drawer.draw(tree, canvasRef.current, "dark", false);
      }, (err: any) => {
        console.error("SmilesDrawer error:", err);
      });
    }
  }, [smiles]);

  return (
    <div className="flex items-center justify-center bg-slate-800/50 rounded-lg p-4 border border-slate-700 h-64 w-full">
      <canvas ref={canvasRef} width={400} height={250} className="w-full h-full object-contain" />
    </div>
  );
};

/**
 * Renders the 3D structure of a molecule using 3Dmol.js.
 */
const Viewer3D = ({ smiles }: { smiles: string }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current || !window.$3Dmol) return;

    if (!viewerRef.current) {
      const config = { backgroundColor: '#0f172a' };
      viewerRef.current = window.$3Dmol.createViewer(containerRef.current, config);
    }

    const viewer = viewerRef.current;
    
    const fetch3D = async () => {
        try {
            const encodedSmiles = encodeURIComponent(smiles);
            const response = await fetch(`https://cactus.nci.nih.gov/chemical/structure/${encodedSmiles}/file?format=sdf&get3d=true`);
            if (response.ok) {
                const sdf = await response.text();
                viewer.clear();
                viewer.addModel(sdf, "sdf");
                viewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } });
                viewer.zoomTo();
                viewer.render();
            }
        } catch (e) {
            // Silently fail for 3D fetch to avoid cluttering console or triggering global handlers
        }
    };

    fetch3D();
  }, [smiles]);

  return (
    <div ref={containerRef} className="w-full h-64 bg-slate-900 rounded-lg border border-slate-700 relative overflow-hidden">
        <div className="absolute top-2 right-2 text-xs text-slate-500 z-10">Interactive 3D</div>
    </div>
  );
};

/**
 * Drug Radar Chart (Spider Plot) for visualizing Molecular Properties.
 */
const DrugRadarChart = ({ properties }: { properties: MoleculeData['properties'] }) => {
    const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);

    // Configuration for axes
    const axes = useMemo(() => [
        { key: 'molecularWeight', label: 'MW', max: 600, limit: 500 },
        { key: 'logP', label: 'LogP', max: 6, limit: 5 },
        { key: 'hbd', label: 'HBD', max: 8, limit: 5 },
        { key: 'hba', label: 'HBA', max: 15, limit: 10 },
        { key: 'tpsa', label: 'TPSA', max: 160, limit: 140 },
        { key: 'rotatableBonds', label: 'RotB', max: 15, limit: 10 },
    ], []);

    const parseValue = (val: string | undefined) => {
        if (!val) return 0;
        const num = parseFloat(val.replace(/[^0-9.-]/g, ''));
        return isNaN(num) ? 0 : num;
    };

    const size = 250;
    const center = size / 2;
    const radius = size / 2 - 40; // Padding for labels

    const polarToCartesian = (angleDeg: number, r: number) => {
        const angleRad = (angleDeg - 90) * (Math.PI / 180);
        return {
            x: center + r * Math.cos(angleRad),
            y: center + r * Math.sin(angleRad),
        };
    };

    // Calculate points for the data polygon
    const dataPoints = axes.map((axis, i) => {
        const value = parseValue(properties?.[axis.key]);
        const normalized = Math.min(Math.max(value / axis.max, 0), 1.1); // Cap slightly above 100%
        return { 
            ...polarToCartesian(i * (360 / axes.length), normalized * radius),
            value,
            label: axis.label,
            fullLabel: axis.key,
            limit: axis.limit
        };
    });

    const dataPath = dataPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x},${p.y}`).join(' ') + 'Z';

    // Calculate background grid (webs)
    const gridLevels = [0.2, 0.4, 0.6, 0.8, 1.0];
    const gridPaths = gridLevels.map(level => {
        return axes.map((_, i) => {
            const pos = polarToCartesian(i * (360 / axes.length), radius * level);
            return `${i === 0 ? 'M' : 'L'} ${pos.x},${pos.y}`;
        }).join(' ') + 'Z';
    });

    // Calculate limit zone (Lipinski Rule of 5 boundary)
    const limitPoints = axes.map((axis, i) => {
         const normalizedLimit = Math.min(axis.limit / axis.max, 1.0);
         return polarToCartesian(i * (360 / axes.length), normalizedLimit * radius);
    });
    const limitPath = limitPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x},${p.y}`).join(' ') + 'Z';

    return (
        <div className="flex flex-col items-center justify-center py-2 relative">
            <svg width={size} height={size} className="overflow-visible">
                {/* Background Grid */}
                {gridPaths.map((path, i) => (
                    <path key={i} d={path} fill="none" stroke="#334155" strokeWidth="1" strokeDasharray="2,2" />
                ))}

                {/* Axes Lines */}
                {axes.map((_, i) => {
                    const pos = polarToCartesian(i * (360 / axes.length), radius);
                    return <line key={i} x1={center} y1={center} x2={pos.x} y2={pos.y} stroke="#334155" strokeWidth="1" />;
                })}

                {/* Limit Zone (Rule of 5 Safe Zone) */}
                <path d={limitPath} fill="none" stroke="#10b981" strokeWidth="1.5" strokeDasharray="4,2" opacity="0.4" />

                {/* Data Polygon */}
                <path d={dataPath} fill="rgba(34, 211, 238, 0.2)" stroke="#22d3ee" strokeWidth="2" className="drop-shadow-[0_0_8px_rgba(34,211,238,0.3)] transition-all duration-500 ease-out" />

                {/* Data Points (Interactive) */}
                {dataPoints.map((p, i) => (
                    <g key={i} 
                       onMouseEnter={() => setHoveredPoint(i)} 
                       onMouseLeave={() => setHoveredPoint(null)}
                       className="cursor-crosshair"
                    >
                        <circle cx={p.x} cy={p.y} r={hoveredPoint === i ? 5 : 3} fill="#020617" stroke="#22d3ee" strokeWidth="2" className="transition-all" />
                        
                        {/* Axis Labels */}
                        {(() => {
                            const labelPos = polarToCartesian(i * (360 / axes.length), radius + 15);
                            return (
                                <text 
                                    x={labelPos.x} 
                                    y={labelPos.y} 
                                    textAnchor="middle" 
                                    dominantBaseline="middle" 
                                    fill={p.value > p.limit ? "#f87171" : "#94a3b8"}
                                    fontSize="10" 
                                    fontWeight="600"
                                    className="font-mono"
                                >
                                    {p.label}
                                </text>
                            );
                        })()}
                    </g>
                ))}
            </svg>

            {/* Tooltip Overlay */}
            {hoveredPoint !== null && (
                <div 
                    className="absolute bg-slate-800 border border-slate-700 rounded px-2 py-1 shadow-xl z-10 pointer-events-none transform -translate-y-8"
                    style={{ 
                        left: dataPoints[hoveredPoint].x, 
                        top: dataPoints[hoveredPoint].y 
                    }}
                >
                    <div className="text-[10px] text-slate-400 font-medium uppercase">{dataPoints[hoveredPoint].fullLabel}</div>
                    <div className={`text-xs font-bold font-mono ${dataPoints[hoveredPoint].value > dataPoints[hoveredPoint].limit ? 'text-red-400' : 'text-cyan-400'}`}>
                        {dataPoints[hoveredPoint].value} <span className="text-slate-500 text-[9px]">/ {dataPoints[hoveredPoint].limit}</span>
                    </div>
                </div>
            )}
            
            <div className="mt-2 flex items-center gap-4 text-[10px]">
                <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                    <span className="text-slate-400">Molecule</span>
                </div>
                <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full border border-emerald-500 border-dashed"></span>
                    <span className="text-emerald-500/80">Rule of 5 Limit</span>
                </div>
            </div>
        </div>
    );
};

/**
 * Message Bubble Component
 */
const ChatMessage: React.FC<{ 
    message: Message, 
    onMoleculeSelect?: (molecules: MoleculeData[]) => void 
}> = ({ message, onMoleculeSelect }) => {
  const isUser = message.role === "user";
  const [showThinking, setShowThinking] = useState(!!message.isStreaming);

  const hasMolecules = !isUser && message.molecules && message.molecules.length > 0;

  useEffect(() => {
      if (message.isStreaming) {
          // If streaming and thoughts appear, auto-open if not already open
          if (message.thoughts && !showThinking) {
              setShowThinking(true);
          }
      } else {
          // If streaming is finished, default to closed
          setShowThinking(false);
      }
  }, [message.isStreaming, message.thoughts]);

  const handleContentClick = (e: React.MouseEvent) => {
      // Avoid triggering if selecting text
      const selection = window.getSelection();
      if (selection && selection.toString().length > 0) return;

      if (hasMolecules && onMoleculeSelect) {
          onMoleculeSelect(message.molecules!);
      }
  };

  return (
    <div className={`flex w-full gap-3 mb-4 ${isUser ? "flex-row-reverse" : "flex-row"} animate-fadeIn group`}>
      <div className={`
        flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center border shadow-sm mt-0.5 transition-colors
        ${isUser 
          ? "bg-cyan-900/20 border-cyan-800/30 text-cyan-400" 
          : message.isError 
            ? "bg-red-900/20 border-red-800/30 text-red-400" 
            : "bg-cyan-900/20 border-cyan-800/30 text-cyan-400"}
      `}>
        <span className="material-symbols-outlined text-[16px]">
          {isUser ? "person" : (message.isError ? "error" : "smart_toy")}
        </span>
      </div>

      <div className={`flex flex-col max-w-[85%] ${isUser ? "items-end" : "items-start"}`}>
        {message.attachments && message.attachments.length > 0 && (
           <div className={`flex flex-wrap gap-2 mb-2 ${isUser ? "justify-end" : "justify-start"}`}>
             {message.attachments.map((att, idx) => (
               <div key={idx} className={`rounded-lg overflow-hidden border ${isUser ? "border-slate-700 bg-slate-800" : "border-slate-800 bg-slate-900"} max-w-[200px] shadow-sm`}>
                 {att.mimeType.startsWith('image/') ? (
                     <img 
                       src={`data:${att.mimeType};base64,${att.data}`} 
                       alt={att.name}
                       className="w-full h-auto object-cover max-h-[150px]"
                     />
                 ) : (
                     <div className="flex items-center gap-3 p-3 text-xs text-slate-300">
                         <div className="w-8 h-8 rounded bg-slate-700 flex items-center justify-center text-slate-400 flex-shrink-0">
                             <span className="material-symbols-outlined text-[18px]">description</span>
                         </div>
                         <span className="truncate max-w-[140px] font-mono">{att.name}</span>
                     </div>
                 )}
               </div>
             ))}
           </div>
        )}
        
        <div 
            onClick={handleContentClick}
            className={`
              relative px-4 py-3 rounded-2xl text-[13px] leading-6 shadow-md
              ${isUser 
                ? "bg-[#1e293b] text-white rounded-tr-sm" 
                : message.isError
                  ? "bg-[#1f0f0f] border border-red-900/50 text-red-200 rounded-tl-sm w-full"
                  : `bg-[#0f172a] border border-[#1e293b] text-white rounded-tl-sm w-full ${hasMolecules ? 'cursor-pointer hover:border-cyan-700/50 hover:bg-[#131d33] hover:shadow-cyan-900/10 transition-all' : ''}`}
            `}
            title={hasMolecules ? "Click to view molecules in Workspace" : undefined}
        >
          
          {hasMolecules && (
             <div className="absolute -top-2.5 -right-2 bg-cyan-950 border border-cyan-800 text-cyan-400 text-[9px] px-1.5 py-0.5 rounded uppercase font-bold tracking-wider opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-sm">
                 View Data
             </div>
          )}

          {!isUser && message.thoughts && !message.isError && (
            <div 
                className="mb-3 border border-slate-800 bg-slate-900/50 rounded-lg overflow-hidden w-full max-w-2xl"
                onClick={(e) => e.stopPropagation()} // Prevent bubble click when interacting with thought toggle
            >
              <button 
                onClick={() => setShowThinking(!showThinking)}
                className="flex items-center justify-between px-3 py-2 text-xs font-mono text-cyan-500/80 hover:text-cyan-400 hover:bg-slate-800/50 transition-colors w-full text-left"
              >
                <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[14px]">
                        {showThinking ? "visibility_off" : "psychology"}
                    </span>
                    <span className="tracking-wide uppercase text-[10px] font-bold flex items-center gap-2">
                        {message.isStreaming ? (
                            <span className="flex items-center gap-1">
                                Thinking <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse inline-block"></span>
                            </span>
                        ) : "Thought Process"}
                    </span>
                </div>
                <span className="material-symbols-outlined text-[14px] opacity-60">
                    {showThinking ? "expand_less" : "expand_more"}
                </span>
              </button>
              
              {showThinking && (
                <div className="px-4 py-3 border-t border-slate-800 bg-[#020617]/50">
                    <div className="prose prose-invert max-w-none font-mono text-slate-400 prose-ul:my-1 prose-li:my-0 text-[11px] leading-5">
                        <ReactMarkdown>{message.thoughts}</ReactMarkdown>
                    </div>
                </div>
              )}
            </div>
          )}

          {message.content && (
              <div className="prose prose-invert max-w-none text-[13px] leading-6 prose-p:text-white prose-pre:bg-[#020617] prose-pre:border prose-pre:border-slate-800 prose-pre:p-3 prose-pre:rounded-lg">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
          )}
          
          {message.isStreaming && !message.content && !message.thoughts && (
             <div className="flex items-center gap-1 text-slate-500 text-xs py-2">
                <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce"></span>
                <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce delay-100"></span>
                <span className="w-1 h-1 rounded-full bg-slate-500 animate-bounce delay-200"></span>
             </div>
          )}
        </div>
      </div>
    </div>
  );
};

const Header = ({ currentModel, onModelChange }: { currentModel: string; onModelChange: (m: string) => void }) => (
  <header className="h-12 border-b border-slate-900 flex items-center justify-between px-5 bg-[#020617]/80 backdrop-blur-md sticky top-0 z-50">
    <div className="flex items-center gap-3">
      <div className="w-6 h-6 rounded bg-gradient-to-br from-cyan-600 to-blue-700 flex items-center justify-center shadow-lg shadow-cyan-900/20">
        <span className="material-symbols-outlined text-white text-[14px]">medication_liquid</span>
      </div>
      <h1 className="font-semibold text-[13px] tracking-wide text-slate-200">Open Drug</h1>
    </div>
    <div className="flex items-center gap-3">
      <div className="relative group">
        <select 
            value={currentModel}
            onChange={(e) => onModelChange(e.target.value)}
            className="appearance-none bg-cyan-950/20 border border-cyan-900/30 text-[10px] font-medium text-cyan-500/80 pl-2.5 pr-6 py-1 rounded cursor-pointer focus:outline-none focus:border-cyan-500/50 transition-all hover:bg-cyan-950/30"
        >
            {MODELS.map(model => (
                <option key={model.id} value={model.id} className="bg-[#0f172a] text-slate-300">
                    {model.name}
                </option>
            ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-1.5 text-cyan-500/80">
            <span className="material-symbols-outlined text-[12px]">expand_more</span>
        </div>
      </div>
    </div>
  </header>
);

const EmptyState = () => (
    <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-600">
        <div className="w-14 h-14 rounded-2xl bg-[#1e293b] border border-slate-800 flex items-center justify-center mb-5 shadow-xl">
           <span className="material-symbols-outlined text-2xl text-slate-500">science</span>
        </div>
        <h3 className="text-slate-300 font-medium mb-2 text-base">Research Assistant Ready</h3>
        <p className="text-[13px] max-w-xs leading-relaxed text-slate-400">
             Specialized in Drug Discovery, Chemistry, Biology, Medicine & AI.
        </p>
    </div>
);

// --- Main Application ---

const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

const App = () => {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'model', content: 'Research Assistant ready. How can I help you today?' }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [moleculeList, setMoleculeList] = useState<MoleculeData[]>([]);
  const [activeMoleculeIndex, setActiveMoleculeIndex] = useState<number>(0);
  const activeMolecule = moleculeList[activeMoleculeIndex] || null;
  const [currentModel, setCurrentModel] = useState<string>("gemini-3-flash-preview");
  const [activeTab, setActiveTab] = useState<'2d' | '3d' | 'properties'>('2d');
  const [leftPanelWidth, setLeftPanelWidth] = useState(40);
  const [isDragging, setIsDragging] = useState(false);
  
  // New State for Interactions
  const [attachments, setAttachments] = useState<File[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
      if (activeMolecule) setActiveTab('2d');
  }, [activeMolecule]);

  // Suppress the default environment error toast for quota exceeded,
  // as we handle it gracefully in the chat UI.
  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
        const message = (event.reason?.message || String(event.reason)).toLowerCase();
        // Check for key terms in the popup message provided by the user
        if (message.includes("quota") || message.includes("429") || message.includes("gemini")) {
            event.preventDefault(); // Suppress global error reporting/toasts
        }
    };
    
    window.addEventListener("unhandledrejection", handleUnhandledRejection);
    return () => window.removeEventListener("unhandledrejection", handleUnhandledRejection);
  }, []);

  const startResizing = useCallback(() => setIsDragging(true), []);
  const stopResizing = useCallback(() => setIsDragging(false), []);
  const resize = useCallback((e: MouseEvent) => {
    if (isDragging) {
      const newWidth = (e.clientX / window.innerWidth) * 100;
      if (newWidth >= 20 && newWidth <= 80) setLeftPanelWidth(newWidth);
    }
  }, [isDragging]);

  useEffect(() => {
    window.addEventListener("mousemove", resize);
    window.addEventListener("mouseup", stopResizing);
    return () => {
      window.removeEventListener("mousemove", resize);
      window.removeEventListener("mouseup", stopResizing);
    };
  }, [resize, stopResizing]);

  // --- Interaction Handlers ---

  // Helper to enforce file limit
  const handleAddFiles = (newFiles: File[]) => {
      const MAX_FILES = 10;
      
      setAttachments(prev => {
          const currentCount = prev.length;
          const remainingSlots = MAX_FILES - currentCount;

          if (remainingSlots <= 0) {
              const isChinese = navigator.language.startsWith('zh');
              setTimeout(() => alert(isChinese 
                ? `单条消息最多只能上传 ${MAX_FILES} 个文件。` 
                : `You can only upload a maximum of ${MAX_FILES} files per message.`), 0);
              return prev;
          }

          if (newFiles.length > remainingSlots) {
              const isChinese = navigator.language.startsWith('zh');
              setTimeout(() => alert(isChinese 
                ? `已达到文件数量上限（${MAX_FILES}个）。仅添加了前 ${remainingSlots} 个文件。` 
                : `File limit of ${MAX_FILES} reached. Only the first ${remainingSlots} files were added.`), 0);
              return [...prev, ...newFiles.slice(0, remainingSlots)];
          }

          return [...prev, ...newFiles];
      });
  };

  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleAddFiles(Array.from(e.target.files));
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const handleMicClick = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert("Speech recognition is not supported in this browser.");
        return;
    }

    if (isListening) {
        return; 
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    
    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onerror = (e: any) => { console.error(e); setIsListening(false); };
    recognition.onresult = (e: any) => {
        const transcript = e.results[0][0].transcript;
        setInput(prev => prev + (prev ? ' ' : '') + transcript);
    };
    
    recognition.start();
  };

  const handleMoleculeDragStart = (e: React.DragEvent, name: string) => {
      e.dataTransfer.setData("text/plain", `@${name}`);
      e.dataTransfer.effectAllowed = "copy";
  };

  // Drag & Drop Handlers
  const handleDragOver = (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      
      // Handle Files
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          handleAddFiles(Array.from(e.dataTransfer.files));
          return;
      }
      
      // Handle Text (e.g. Molecule Names from Workspace)
      const textData = e.dataTransfer.getData("text/plain");
      if (textData) {
          setInput(prev => {
             const prefix = prev && !prev.endsWith(' ') ? ' ' : '';
             return prev + prefix + textData;
          });
      }
  };

  // Paste Handler
  const handlePaste = (e: React.ClipboardEvent) => {
      if (e.clipboardData.files && e.clipboardData.files.length > 0) {
          e.preventDefault();
          handleAddFiles(Array.from(e.clipboardData.files));
      }
  };

  const handleSend = async () => {
    if ((!input.trim() && attachments.length === 0) || isLoading) return;

    if (input.trim().toLowerCase() === "clear") {
        setMessages([{ role: 'model', content: 'Research Assistant ready.' }]);
        setInput("");
        setMoleculeList([]);
        setAttachments([]);
        return;
    }

    const processedAttachments: Attachment[] = [];
    if (attachments.length > 0) {
        try {
            const results = await Promise.all(attachments.map(async (file) => ({
                name: file.name,
                mimeType: file.type,
                data: await fileToBase64(file)
            })));
            processedAttachments.push(...results);
        } catch (error) {
            console.error("Error processing files:", error);
            return;
        }
    }

    const userMsg: Message = { 
        role: "user", 
        content: input,
        attachments: processedAttachments.length > 0 ? processedAttachments : undefined
    };
    
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setAttachments([]); 
    if (fileInputRef.current) fileInputRef.current.value = "";
    setIsLoading(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || "" });
      
      const history = messages.map(m => {
          const parts: any[] = [];
          if (m.attachments) {
              m.attachments.forEach(att => {
                  parts.push({
                      inlineData: {
                          mimeType: att.mimeType,
                          data: att.data
                      }
                  });
              });
          }
          if (m.content) {
              parts.push({ text: m.content });
          }
          return {
              role: m.role,
              parts: parts
          };
      });
      
      const currentParts: any[] = [];
      if (userMsg.attachments) {
          userMsg.attachments.forEach(att => {
              currentParts.push({
                  inlineData: {
                      mimeType: att.mimeType,
                      data: att.data
                  }
              });
          });
      }
      if (userMsg.content) {
          currentParts.push({ text: userMsg.content });
      }

      setMessages(prev => [...prev, { role: "model", content: "", thoughts: "", isStreaming: true }]);

      const result = await ai.models.generateContentStream({
         model: currentModel,
         contents: [...history, { role: 'user', parts: currentParts }],
         config: {
             systemInstruction: SYSTEM_INSTRUCTION,
             tools: [{ googleSearch: {} }],
             thinkingConfig: { thinkingBudget: 4096 }
         }
      });

      let fullText = "";
      
      for await (const chunk of result) {
          fullText += (chunk.text || "");
          
          let thoughts = "";
          let content = fullText;

          const openTag = "<thinking>";
          const closeTag = "</thinking>";
          const openIndex = fullText.indexOf(openTag);
          
          if (openIndex !== -1) {
              const closeIndex = fullText.indexOf(closeTag);
              if (closeIndex !== -1) {
                  thoughts = fullText.substring(openIndex + openTag.length, closeIndex).trim();
                  content = fullText.substring(closeIndex + closeTag.length).trim();
              } else {
                  thoughts = fullText.substring(openIndex + openTag.length).trim();
                  content = "";
              }
          }

          setMessages(prev => {
              const newMessages = [...prev];
              const lastMsg = newMessages[newMessages.length - 1];
              lastMsg.content = content;
              lastMsg.thoughts = thoughts;
              return newMessages;
          });
      }

      const jsonMatch = fullText.match(/```json\s*({[\s\S]*?"molecules"[\s\S]*?})\s*```/);
      if (jsonMatch) {
          try {
              const data = JSON.parse(jsonMatch[1]);
              if (Array.isArray(data.molecules)) {
                  setMoleculeList(data.molecules);
                  setActiveMoleculeIndex(0);
                  const cleanContent = fullText.replace(/<thinking>[\s\S]*?<\/thinking>/, "").replace(jsonMatch[0], "").trim();
                  setMessages(prev => {
                      const newMessages = [...prev];
                      const lastMsg = newMessages[newMessages.length - 1];
                      lastMsg.content = cleanContent;
                      lastMsg.molecules = data.molecules; // Store molecules in message
                      return newMessages;
                  });
              }
          } catch (e) { console.error(e); }
      }

      setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].isStreaming = false;
          return newMessages;
      });

    } catch (error: any) {
      // Handle quota and other errors elegantly in the UI
      let errorMessage = error.message || "An unexpected error occurred.";
      const errStr = errorMessage.toLowerCase();

      // Check for common quota/rate limit indicators
      if (errStr.includes("429") || errStr.includes("quota") || errStr.includes("resource has been exhausted")) {
         const isChinese = navigator.language.startsWith('zh');
         errorMessage = isChinese 
            ? "⚠️ API 额度已耗尽。当前模型的使用限制已达到，请在右上角下拉菜单中切换到其他模型（例如 Gemini 2.5 Flash）。"
            : "⚠️ API quota exhausted. The current model's usage limit has been reached. Please switch to another model (e.g., Gemini 2.5 Flash) using the dropdown menu in the top right.";
      }

      setMessages(prev => {
          const newMessages = [...prev];
          const lastMsg = newMessages[newMessages.length - 1];
          lastMsg.content = errorMessage;
          lastMsg.isError = true;
          lastMsg.isStreaming = false;
          return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleMessageClick = (molecules: MoleculeData[]) => {
      if (molecules && molecules.length > 0) {
          setMoleculeList(molecules);
          setActiveMoleculeIndex(0);
      }
  };

  return (
    <div className={`flex flex-col h-screen bg-[#020617] text-slate-200 font-sans ${isDragging ? 'cursor-col-resize select-none' : ''}`}>
      <Header currentModel={currentModel} onModelChange={setCurrentModel} />
      
      <main className="flex-1 flex overflow-hidden">
        
        {/* Chat Panel */}
        <div style={{ width: `${leftPanelWidth}%` }} className="flex flex-col border-r border-slate-900 relative bg-[#020617] min-w-[300px]">
          <div className="flex-1 overflow-y-auto p-5 custom-scrollbar">
            {messages.length === 0 ? <EmptyState /> : messages.map((msg, idx) => (
                <ChatMessage 
                    key={idx} 
                    message={msg} 
                    onMoleculeSelect={handleMessageClick}
                />
            ))}
            <div ref={bottomRef} className="h-4" />
          </div>

          <div className="p-6 bg-[#020617]">
            <div className="max-w-4xl mx-auto w-full">
              
              {/* Hidden File Input */}
              <input 
                type="file" 
                multiple
                ref={fileInputRef} 
                onChange={handleFileChange} 
                className="hidden" 
              />

              <div 
                  className={`relative flex flex-col bg-[#1e293b] rounded-2xl border transition-all shadow-lg shadow-black/20 
                  ${isDragOver ? 'border-cyan-400 bg-cyan-900/20 ring-1 ring-cyan-400/20' : 'border-slate-800 focus-within:border-cyan-400/50 focus-within:ring-1 focus-within:ring-cyan-400/20'}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
              >
                
                {/* Attachment UI */}
                {attachments.length > 0 && (
                   <div className="w-full px-4 pt-4 pb-2 flex flex-wrap gap-2 max-h-[200px] overflow-y-auto custom-scrollbar z-10">
                       <div className="flex-shrink-0 flex items-center justify-center bg-slate-800/80 rounded-lg px-3 h-8 text-[10px] text-slate-400 font-mono border border-slate-700">
                           {attachments.length}/10
                       </div>
                       {attachments.map((file, idx) => (
                           <div key={idx} className="flex-shrink-0 bg-cyan-950/90 border border-cyan-800/50 text-cyan-200 text-xs pl-3 pr-2 h-8 rounded-lg flex items-center gap-2 animate-fadeIn backdrop-blur-sm shadow-sm max-w-[220px]">
                               <span className="material-symbols-outlined text-[14px]">{file.type.startsWith('image/') ? 'image' : 'description'}</span>
                               <span className="truncate font-mono">{file.name}</span>
                               <button onClick={() => removeAttachment(idx)} className="hover:text-white hover:bg-cyan-800/50 rounded p-0.5 transition-colors ml-1 flex items-center">
                                   <span className="material-symbols-outlined text-[14px]">close</span>
                               </button>
                           </div>
                       ))}
                   </div>
                )}
                
                {/* Drag Overlay Text (Optional) */}
                {isDragOver && (
                    <div className="absolute inset-0 z-20 flex items-center justify-center bg-cyan-950/40 rounded-2xl backdrop-blur-sm pointer-events-none">
                        <div className="text-cyan-400 font-medium flex items-center gap-2">
                             <span className="material-symbols-outlined text-2xl">upload_file</span>
                             Drop files here
                        </div>
                    </div>
                )}

                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSend();
                      }
                  }}
                  onPaste={handlePaste}
                  placeholder="Ask Open Drug"
                  className={`w-full bg-transparent border-none focus:ring-0 pl-6 pr-36 text-[15px] text-slate-200 placeholder-slate-500/60 resize-none ${attachments.length > 0 ? 'min-h-[84px]' : 'min-h-[56px]'} max-h-[200px] custom-scrollbar leading-relaxed outline-none py-5`}
                  rows={1}
                />
                
                <div className="absolute bottom-2 right-2 flex items-center gap-1.5">
                    {/* Microphone Button */}
                    <button 
                        onClick={handleMicClick}
                        className={`w-9 h-9 rounded-xl transition-all flex items-center justify-center group ${isListening ? 'text-red-400 bg-red-900/20 animate-pulse' : 'text-cyan-400/70 hover:text-cyan-400 hover:bg-cyan-900/10'}`} 
                        title="Voice Input"
                    >
                        <span className="material-symbols-outlined text-[20px] group-hover:scale-105 transition-transform">
                            {isListening ? 'mic_off' : 'mic'}
                        </span>
                    </button>

                    {/* File Upload Button */}
                    <button 
                        onClick={handleFileClick}
                        className="w-9 h-9 rounded-xl text-cyan-400/70 hover:text-cyan-400 hover:bg-cyan-900/10 transition-colors flex items-center justify-center group" 
                        title="Upload File"
                    >
                        <span className="material-symbols-outlined text-[20px] group-hover:scale-105 transition-transform">add_circle</span>
                    </button>

                    {/* Send Button */}
                    <button 
                        onClick={handleSend}
                        disabled={(!input.trim() && attachments.length === 0) || isLoading}
                        className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-200 ${input.trim() || attachments.length > 0 ? "bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 shadow-sm shadow-cyan-900/20" : "text-slate-600 cursor-not-allowed"}`}
                        title="Send"
                    >
                        <span className="material-symbols-outlined text-[20px]">arrow_upward</span>
                    </button>
                </div>
              </div>
            </div>
            <div className="text-center mt-4">
                 <p className="text-[11px] text-slate-500 font-medium">AI can make mistakes. Please verify scientific data.</p>
            </div>
          </div>
        </div>

        {/* Resizer */}
        <div 
          className="w-1 bg-[#020617] hover:bg-cyan-600/50 cursor-col-resize flex-shrink-0 transition-colors z-20 flex items-center justify-center group border-l border-slate-900"
          onMouseDown={startResizing}
        >
             <div className="h-8 w-0.5 bg-slate-800 group-hover:bg-cyan-400 rounded-full transition-colors"></div>
        </div>

        {/* Workspace Panel */}
        <div style={{ width: `${100 - leftPanelWidth}%` }} className="bg-[#020617] flex flex-col z-10 min-w-[300px]">
           <div className="px-5 py-3 border-b border-slate-900 flex items-center justify-between bg-[#020617]">
              <h2 className="font-medium text-slate-300 text-[13px] flex items-center gap-2">
                 <span className="material-symbols-outlined text-slate-500 text-[16px]">chemistry</span>
                 Workspace
              </h2>
              {activeMolecule && <span className="text-[10px] font-mono bg-cyan-950/50 text-cyan-400 border border-cyan-900/50 px-2 py-0.5 rounded tracking-wide">ACTIVE</span>}
           </div>

           <div className="flex-1 overflow-y-auto custom-scrollbar bg-[#020617]">
              {activeMolecule ? (
                  <div className="p-5 flex flex-col min-h-full">
                      {moleculeList.length > 1 && (
                        <div className="mb-5 flex flex-wrap gap-2 max-h-[160px] overflow-y-auto custom-scrollbar">
                            {moleculeList.map((mol, idx) => (
                                <button key={idx} draggable={true} onDragStart={(e) => handleMoleculeDragStart(e, mol.name)} onClick={() => setActiveMoleculeIndex(idx)} className={`px-3 py-1.5 rounded-lg text-[11px] font-medium transition-all border cursor-grab active:cursor-grabbing ${idx === activeMoleculeIndex ? 'bg-cyan-900/30 text-cyan-400 border-cyan-500/50' : 'bg-[#0f172a] text-slate-400 border-slate-800'}`}>
                                    {mol.name.substring(0, 15)}
                                </button>
                            ))}
                        </div>
                      )}

                      <div className="mb-5">
                          <h3 className="text-base font-bold text-white mb-2">{activeMolecule.name}</h3>
                          <code className="text-[10px] bg-[#0f172a] px-2 py-1.5 rounded text-cyan-200/80 break-all border border-slate-800 font-mono w-full block mb-2">{activeMolecule.smiles}</code>
                          <p className="text-[13px] text-slate-400 leading-relaxed border-l-2 border-slate-800 pl-3">{activeMolecule.description}</p>
                      </div>

                      <div className="border-b border-slate-800 flex gap-6 mb-5">
                          {['2d', '3d', 'properties'].map((tab) => (
                              <button key={tab} onClick={() => setActiveTab(tab as any)} className={`pb-2 text-[11px] font-bold uppercase tracking-wide transition-colors relative ${activeTab === tab ? 'text-cyan-400' : 'text-slate-500 hover:text-slate-300'}`}>
                                  {tab === '2d' ? '2D Structure' : tab === '3d' ? '3D Structure' : 'Properties'}
                                  {activeTab === tab && <span className="absolute bottom-0 left-0 w-full h-0.5 bg-cyan-400 rounded-t-full"></span>}
                              </button>
                          ))}
                      </div>

                      <div className="flex-1 min-h-[350px]">
                          {activeTab === '2d' && <Viewer2D smiles={activeMolecule.smiles} />}
                          {activeTab === '3d' && <Viewer3D smiles={activeMolecule.smiles} />}
                          {activeTab === 'properties' && (
                              <div className="bg-[#0f172a] rounded-xl border border-slate-800 overflow-hidden p-4">
                                  {/* Interactive Radar Chart */}
                                  <div className="mb-6 flex justify-center bg-slate-900/50 rounded-xl p-4 border border-slate-800/50">
                                      <DrugRadarChart properties={activeMolecule.properties} />
                                  </div>
                                  
                                  <div className="space-y-4">
                                      <h4 className="text-[11px] font-bold uppercase tracking-widest text-slate-500 border-b border-slate-800 pb-2">Detailed Properties</h4>
                                      <div className="grid grid-cols-2 gap-4">
                                          {Object.entries(activeMolecule.properties || {}).map(([key, value]) => (
                                              <div key={key} className="bg-slate-800/30 p-3 rounded-lg border border-slate-800 flex flex-col">
                                                  <span className="text-[10px] text-slate-500 uppercase font-medium mb-1">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                                                  <span className="text-[13px] font-mono text-cyan-300">{value}</span>
                                              </div>
                                          ))}
                                      </div>
                                  </div>
                              </div>
                          )}
                      </div>
                  </div>
              ) : (
                  <div className="h-full flex flex-col items-center justify-center text-slate-700 opacity-60">
                      <span className="material-symbols-outlined text-4xl mb-3">hub</span>
                      <p className="text-[11px] font-medium uppercase tracking-widest">No Active Molecule</p>
                  </div>
              )}
           </div>
        </div>
      </main>
    </div>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);