import React, { useState, useRef, useEffect } from 'react';
import Header from './components/Header';
import AnalysisResults from './components/AnalysisResults';
import Methodology from './components/Methodology';
import Blog from './components/Blog';
import History from './components/History';
import AdminLogin from './components/AdminLogin';
import HomeFeatures from './components/HomeFeatures';
import { analyzeVideo } from './services/gemini';
import { downloadVideo } from './services/downloader';
import { saveSubmission, getSubmission, uploadFile } from './services/api';
import { AnalysisResult, AnalysisStatus, VideoMetadata } from './types';
import { UploadCloud, Youtube, Loader2, PlayCircle, AlertCircle, Link as LinkIcon, CheckCircle2, RefreshCw, Hash, Lock } from 'lucide-react';

const LimitModal = ({ onClose }: { onClose: () => void }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-in fade-in duration-300">
    <div className="bg-slate-900 border border-indigo-500/30 rounded-2xl max-w-md w-full shadow-2xl overflow-hidden p-8 text-center">
      <div className="w-16 h-16 bg-indigo-500/10 rounded-full flex items-center justify-center mx-auto mb-6 ring-1 ring-indigo-500/50">
        <Lock className="w-8 h-8 text-indigo-400" />
      </div>
      <h3 className="text-2xl font-bold text-white mb-2">Daily Limit Reached</h3>
      <p className="text-slate-400 mb-6">
        This beta demo is limited to <strong className="text-indigo-400">5 submissions per day</strong> to maintain service quality.
      </p>
      <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-800 mb-6">
        <p className="text-sm text-slate-300 font-medium mb-1">Need higher limits?</p>
        <p className="text-xs text-slate-500">Contact us for enterprise API access:</p>
        <a href="mailto:admin@frametruth.com" className="text-indigo-400 hover:text-indigo-300 font-mono text-sm mt-1 block underline">
          admin@frametruth.com
        </a>
      </div>
      <button 
        onClick={onClose}
        className="w-full bg-slate-800 hover:bg-slate-700 text-white font-medium py-2.5 rounded-lg transition-colors ring-1 ring-slate-700"
      >
        Close
      </button>
    </div>
  </div>
);

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<'home' | 'methodology' | 'blog' | 'admin'>('home');
  const [adminAuth, setAdminAuth] = useState<{ user: string; pass: string } | null>(null);
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [videoMeta, setVideoMeta] = useState<VideoMetadata | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'upload' | 'link'>('link');
  const [linkUrl, setLinkUrl] = useState('');
  // Store backend filename for DB saving
  const [backendFilename, setBackendFilename] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll when complete
  useEffect(() => {
    if (status === 'complete') {
        setTimeout(() => {
            document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
  }, [status]);

  useEffect(() => {
    // Check route
    if (window.location.pathname === '/admin') {
        setCurrentPage('admin');
    }

    // Load submission from URL ID
    const params = new URLSearchParams(window.location.search);
    const id = params.get('id');
    if (id) {
        loadSubmission(id);
    }

    // Handle browser navigation
    window.onpopstate = () => {
         if (window.location.pathname === '/admin') {
            setCurrentPage('admin');
         } else {
             setCurrentPage('home');
         }
    };
  }, []);

  const loadSubmission = async (id: string) => {
      try {
          setStatus('processing');
          setLoadingMessage('Retrieving Submission...');
          const data = await getSubmission(id);
          
          setVideoMeta({
              name: "Archived Submission",
              url: data.video_url,
              size: 0,
              type: "video/mp4"
          });
          setResult(data.analysis_result);
          setSubmissionId(data.id);
          setStatus('complete');
      } catch(e) {
          setError("Could not load submission. ID may be invalid.");
          setStatus('idle');
      }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Basic validation
    if (file.size > 50 * 1024 * 1024) { // 50MB limit for browser performance
      setError("File size too large. For this web demo, please use videos under 50MB.");
      return;
    }

    setVideoMeta({
      name: file.name,
      size: file.size,
      type: file.type,
      url: URL.createObjectURL(file)
    });
    setError(null);

    // Upload to backend for persistence and analysis
    setStatus('uploading');
    try {
        const uploadRes = await uploadFile(file);
        setBackendFilename(uploadRes.filename);
        // Pass filename AND original name
        processVideo(file, uploadRes.filename, "File: " + file.name);
    } catch (e) {
        console.error("Upload failed", e);
        setError("Failed to upload video for analysis.");
        setStatus('idle');
    }
  };

  const processVideo = async (file: File, filenameForDb?: string, sourceInfo?: string) => {
    try {
      if (!filenameForDb) {
          throw new Error("File upload failed or file not found on server.");
      }

      setStatus('processing');
      setLoadingMessage('Analyzing Frames on Secure Backend...');
      
      // Call Backend Analysis (which also saves the submission)
      // Pass sourceInfo (URL or Filename) so backend can save it
      const { result: analysis, submission_id } = await analyzeVideo(filenameForDb, sourceInfo || '');
      
      setResult(analysis);
      setSubmissionId(submission_id);
      
      // Update URL without reload
      window.history.pushState({}, '', `?id=${submission_id}`);

      setStatus('complete');
    } catch (err: any) {
      setError(err.message || "An unknown error occurred.");
      setStatus('error');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
        if (file && file.type.startsWith('video/')) {
            if (file.size > 50 * 1024 * 1024) {
                setError("File size too large. Please use videos under 50MB.");
                return;
            }
            setVideoMeta({
                name: file.name,
                size: file.size,
                type: file.type,
                url: URL.createObjectURL(file)
            });
            setError(null);
            
            // Upload
            setStatus('uploading');
            uploadFile(file).then(res => {
                 setBackendFilename(res.filename);
                 processVideo(file, res.filename, "File: " + file.name);
            }).catch(() => {
                 setError("Failed to upload video.");
                 setStatus('idle');
            });
            
        } else {
            setError("Please drop a valid video file.");
        }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleLinkSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!linkUrl.trim()) return;
    
    setError(null);
    setVideoMeta(null);
    setResult(null);
    setSubmissionId(null);
    setBackendFilename(null);
    setStatus('downloading');

    try {
        // Use the new downloader service that replicates yt-dlp logic
        // It returns backendFilename if available, and META
        const { file, meta, backendFilename: bFilename } = await downloadVideo(linkUrl, {
            onProgress: (msg) => setLoadingMessage(msg)
        });

        if (bFilename) setBackendFilename(bFilename);

        setVideoMeta({
            name: meta.title || meta.name || 'downloaded_video.mp4', // Use TITLE if available
            size: meta.size || 0,
            type: meta.type || 'video/mp4',
            url: meta.url || ''
        });

        // Proceed to analysis. Pass URL as sourceInfo.
        processVideo(file, bFilename, linkUrl);

    } catch (err: any) {
        console.error(err);
        setStatus('error');
        // Clean up error message for UI
        let msg = err.message || "Download failed.";
        if (msg.includes("Details:")) {
            // If it's a multi-part error, try to find the most relevant part
            if (msg.includes("YouTube")) msg = msg.split("YouTube")[1];
            else if (msg.includes("Cobalt")) msg = "The video could not be extracted. It may be private or geo-blocked.";
        }
        setError(err.message); // Keep full message for now as it has useful details
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 font-sans selection:bg-cyan-500/30">
      <Header onNavigate={setCurrentPage} currentPage={currentPage} />

      {/* Beta Banner */}
      <div className="bg-indigo-900/50 text-center py-2 px-4 text-xs font-medium text-indigo-200 border-b border-indigo-800/50">
        <span className="bg-indigo-500 text-white px-1.5 py-0.5 rounded text-[10px] mr-2 uppercase tracking-widest">Beta</span>
        Limited to 5 submissions per day per IP. For enterprise access, email <a href="mailto:admin@frametruth.com" className="text-white hover:underline">admin@frametruth.com</a>
      </div>

      {currentPage === 'methodology' ? (
        <Methodology />
      ) : currentPage === 'blog' ? (
        <Blog />
      ) : currentPage === 'admin' ? (
        !adminAuth ? (
            <AdminLogin onLogin={(user, pass) => setAdminAuth({ user, pass })} />
        ) : (
            <History adminAuth={adminAuth} onLogout={() => setAdminAuth(null)} />
        )
      ) : (
        <main className="max-w-4xl mx-auto px-4 py-12">
          
          {/* Intro Section - SEO Optimized */}
          <div className="text-center mb-12 space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
              Free <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">AI Video Detector</span> & Deepfake Checker
            </h1>
            <p className="text-slate-400 max-w-2xl mx-auto text-lg">
              <strong>Is this video real or fake?</strong> Verify video authenticity instantly. 
              Our advanced <strong className="text-slate-200">AI video authenticity checker</strong> works for YouTube, TikTok, and social media clips using forensic trajectory analysis.
            </p>
          </div>

          {/* Input Container */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-1 shadow-xl backdrop-blur-sm mb-12 transition-all duration-500">
            <div className="grid grid-cols-2 gap-1 mb-1 p-1 bg-slate-950/50 rounded-xl">
              <button 
                onClick={() => { setActiveTab('upload'); setError(null); }}
                className={`py-3 rounded-lg text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2 ${activeTab === 'upload' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
              >
                <UploadCloud className="w-4 h-4" />
                Upload File
              </button>
              <button 
                onClick={() => { setActiveTab('link'); setError(null); }}
                className={`py-3 rounded-lg text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2 ${activeTab === 'link' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
              >
                <Youtube className="w-4 h-4" />
                Social / Web Link
              </button>
            </div>

            <div className="p-6 md:p-10 min-h-[300px] flex flex-col justify-center">
              {activeTab === 'upload' ? (
                <div 
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  className={`border-2 border-dashed rounded-xl h-64 flex flex-col items-center justify-center transition-all cursor-pointer group relative overflow-hidden
                    ${(status === 'uploading' || status === 'processing' || status === 'downloading') ? 'border-cyan-500/30 bg-cyan-500/5 pointer-events-none' : 'border-slate-700 hover:border-cyan-500/50 hover:bg-slate-800/50'}
                  `}
                  onClick={(status === 'idle' || status === 'complete' || status === 'error') ? triggerFileInput : undefined}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="video/*" 
                    onChange={handleFileChange} 
                  />
                  
                  {(status === 'idle' || status === 'complete' || status === 'error') ? (
                    <>
                      <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform group-hover:bg-slate-700 shadow-lg shadow-black/50">
                        <UploadCloud className="w-8 h-8 text-cyan-400" />
                      </div>
                      <p className="text-lg font-medium text-slate-300">Click to upload or drag video here</p>
                      <p className="text-sm text-slate-500 mt-2">Supports MP4, MOV, WEBM (Max 50MB)</p>
                    </>
                  ) : (
                    <div className="flex flex-col items-center z-10">
                      <div className="relative">
                        <Loader2 className="w-12 h-12 text-cyan-500 animate-spin" />
                        <div className="absolute inset-0 bg-cyan-500 blur-xl opacity-20 animate-pulse"></div>
                      </div>
                      <p className="mt-4 text-cyan-400 font-medium animate-pulse tracking-wide">
                        {status === 'downloading' ? 'Fetching Video...' : (status === 'processing' ? 'Analyzing...' : 'Processing...')}
                      </p>
                      <p className="text-xs text-slate-500 mt-2 uppercase tracking-widest opacity-70">
                        {loadingMessage || 'Please wait...'}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <form onSubmit={handleLinkSubmit} className="h-64 flex flex-col items-center justify-center w-full max-w-lg mx-auto">
                  {(status === 'downloading' || status === 'processing' || status === 'uploading') ? (
                     <div className="flex flex-col items-center z-10">
                      <div className="relative">
                        <Loader2 className="w-12 h-12 text-cyan-500 animate-spin" />
                        <div className="absolute inset-0 bg-cyan-500 blur-xl opacity-20 animate-pulse"></div>
                      </div>
                      <p className="mt-4 text-cyan-400 font-medium animate-pulse tracking-wide">
                         {status === 'downloading' ? 'Downloading Stream' : 'Analyzing Content'}
                      </p>
                      <p className="text-xs text-slate-500 mt-2 uppercase tracking-widest opacity-70 font-mono">
                         {loadingMessage || 'Initializing Engine...'}
                      </p>
                    </div>
                  ) : (
                  <div className="w-full space-y-6">
                    <div className="text-center space-y-2">
                        <h3 className="text-lg font-medium text-white">Analyze Video from URL</h3>
                        <p className="text-slate-400 text-sm">Paste a link from YouTube, Instagram, Tiktok, Facebook, or any direct video link</p>
                    </div>
                    
                    <div className="flex gap-2">
                          <div className="relative flex-1 group">
                              <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
                                  <LinkIcon className="w-4 h-4 text-slate-500 group-focus-within:text-cyan-500 transition-colors" />
                              </div>
                              <input 
                                  type="url" 
                                  value={linkUrl}
                                  onChange={(e) => setLinkUrl(e.target.value)}
                                  placeholder="Paste video link here..." 
                                  className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-10 pr-4 py-3 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all placeholder:text-slate-700"
                              />
                          </div>
                          <button 
                              type="submit" 
                              disabled={!linkUrl}
                              className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 rounded-lg font-medium transition-all shadow-lg shadow-cyan-900/20 active:scale-95"
                          >
                              Scan
                          </button>
                      </div>
                      
                    <div className="flex flex-wrap justify-center gap-4 text-xs text-slate-500 opacity-60">
                        <span className="flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> YouTube</span>
                        <span className="flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Instagram</span>
                        <span className="flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Tiktok</span>
                        <span className="flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Facebook</span>
                    </div>
                  </div>
                  )}
                </form>
              )}
            </div>
          </div>
          
          {/* Keyword Rich Features Section */}
          {!result && <HomeFeatures />}

          {/* Error Message / Limit Modal */}
            {error && (
                error.includes("Daily submission limit") ? (
                    <LimitModal onClose={() => setError(null)} />
                ) : (
                    <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-lg mb-8 flex items-start gap-3 animate-fade-in">
                        <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                        <div className="flex-1">
                            <p className="font-semibold">Error</p>
                            <p className="text-sm opacity-90 break-words">{error}</p>
                            {activeTab === 'link' && (
                                <button onClick={handleLinkSubmit} className="mt-2 text-xs bg-rose-500/20 hover:bg-rose-500/30 px-2 py-1 rounded flex items-center gap-1 transition-colors">
                                    <RefreshCw className="w-3 h-3" /> Retry Link
                                </button>
                            )}
                        </div>
                    </div>
                )
            )}

          {/* Results Section */}
          {result && status === 'complete' && (
            <div id="results" className="scroll-mt-24 animate-in fade-in slide-in-from-bottom-4 duration-700">
               {videoMeta && (
                  <div className="flex items-center gap-4 mb-6 p-4 bg-slate-900/30 rounded-lg border border-slate-800/50 backdrop-blur-sm">
                      <div className="w-20 h-20 bg-black rounded-lg overflow-hidden relative flex-shrink-0 border border-slate-800 group">
                          <video src={videoMeta.url} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity" />
                          <div className="absolute inset-0 flex items-center justify-center bg-black/20 pointer-events-none">
                              <PlayCircle className="w-8 h-8 text-white/90 drop-shadow-md" />
                          </div>
                      </div>
                      <div className="overflow-hidden">
                          <h3 className="font-medium text-white text-lg truncate pr-4">{videoMeta.name}</h3>
                          <p className="text-xs text-slate-400 font-mono mt-1 flex items-center gap-2">
                              <span className="bg-slate-800 px-1.5 py-0.5 rounded">{(videoMeta.size / (1024*1024)).toFixed(1)} MB</span>
                              <span>{videoMeta.type.split('/')[1]?.toUpperCase() || 'VIDEO'}</span>
                          </p>
                      </div>
                  </div>
               )}
              
              {/* Submission ID Badge */}
              {submissionId && (
                  <div className="flex justify-center mb-6 animate-fade-in">
                      <div className="bg-slate-900 border border-cyan-500/30 rounded-full px-4 py-1 text-sm text-cyan-400 flex items-center gap-2 font-mono">
                          <Hash className="w-3 h-3" />
                          Submission ID: <span className="font-bold select-all">{submissionId}</span>
                      </div>
                  </div>
              )}

              <AnalysisResults result={result} submissionId={submissionId} />
            </div>
          )}
        </main>
      )}

      {/* Footer */}
      <footer className="border-t border-slate-900 py-8 mt-12 text-center">
        <p className="text-slate-600 text-sm mb-2">
          Based on the paper "AI-Generated Video Detection via Perceptual Straightening" (Internò et al., 2025).
          <br/>
          Powered by Gemini for semantic analysis.
        </p>
        <div className="text-xs text-slate-700 mt-4">
           Detect AI Video | Deepfake Detector | Video Forensics Tool
        </div>
      </footer>
    </div>
  );
};

export default App;
