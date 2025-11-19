import React, { useEffect, useState } from 'react';
import { getSubmissions } from '../services/api';
import { PlayCircle, Calendar, AlertCircle, CheckCircle, Globe, Lock, Search, ChevronLeft, ChevronRight } from 'lucide-react';

interface HistoryProps {
  adminAuth: { user: string; pass: string };
  onLogout: () => void;
}

const History: React.FC<HistoryProps> = ({ adminAuth, onLogout }) => {
  const [submissions, setSubmissions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const limit = 50;

  useEffect(() => {
    loadHistory();
  }, [currentPage, searchQuery]);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSubmissions(adminAuth.user, adminAuth.pass, searchQuery, currentPage, limit);
      console.log('API Response:', data);
      
      // Handle both old (array) and new (paginated object) response formats
      if (Array.isArray(data)) {
        // Old format - direct array
        console.log('Using old array format');
        setSubmissions(data);
        setTotal(data.length);
        setTotalPages(1);
      } else if (data.submissions && Array.isArray(data.submissions)) {
        // New format - paginated object
        console.log('Using new paginated format');
        setSubmissions(data.submissions);
        setTotal(data.total || 0);
        setTotalPages(data.pages || 1);
      } else {
        console.error('Unexpected response format:', data);
        throw new Error('Invalid response format from server. Check console for details.');
      }
    } catch (e: any) {
      console.error('History Load Error:', e);
      setError(e.message || "Access Denied or Server Error");
      setSubmissions([]);
      setTotal(0);
      setTotalPages(1);
    }
    setLoading(false);
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setCurrentPage(1); // Reset to first page on new search
  };

  const formatDate = (isoString: string) => {
    return new Date(isoString).toLocaleString();
  };

  const viewSubmission = (id: string) => {
      // Full refresh to reset app state properly or navigation if we had router
      window.location.href = `/?id=${id}`;
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-12 animate-fade-in">
       <div className="mb-12 text-center relative">
        <button onClick={onLogout} className="absolute right-0 top-0 text-xs text-slate-500 hover:text-white flex items-center gap-1">
            <Lock className="w-3 h-3" /> Logout
        </button>
        <h2 className="text-3xl font-bold text-white mb-4">Submission <span className="text-cyan-400">History</span></h2>
        <p className="text-slate-400">Restricted Admin Area • Total: {total} submissions</p>
      </div>

      {/* Search Bar */}
      <div className="mb-6">
        <div className="relative max-w-md mx-auto">
          <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
            <Search className="w-4 h-4 text-slate-500" />
          </div>
          <input 
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search by ID, IP, URL, filename..." 
            className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-10 pr-4 py-3 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all placeholder:text-slate-600"
          />
        </div>
      </div>

      {loading ? (
          <div className="text-center text-slate-500 py-20">Authenticating...</div>
      ) : error ? (
          <div className="text-center text-rose-500 py-20 border border-rose-900/30 rounded-xl bg-rose-500/5">
              <AlertCircle className="w-8 h-8 mx-auto mb-2" />
              {error}
          </div>
      ) : submissions.length === 0 ? (
          <div className="text-center text-slate-500 py-20 border border-slate-800 rounded-xl bg-slate-900/50">
            {searchQuery ? `No submissions found matching "${searchQuery}"` : 'No submissions found.'}
          </div>
      ) : (
        <>
          <div className="grid gap-4">
              {submissions.map((sub) => (
                  <div key={sub.id} onClick={() => viewSubmission(sub.id)} className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl flex flex-col md:flex-row items-center gap-6 hover:bg-slate-900 hover:border-cyan-500/30 transition-all cursor-pointer group">
                      {/* Thumbnail Placeholder / Video Preview */}
                      <div className="w-full md:w-40 h-24 bg-black rounded-lg overflow-hidden relative border border-slate-800 group-hover:border-cyan-500/50 flex-shrink-0">
                          <video src={sub.video_url} className="w-full h-full object-cover opacity-60" />
                          <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                             <PlayCircle className="w-8 h-8 text-white/80" />
                          </div>
                      </div>
                      
                      <div className="flex-1 w-full">
                          <div className="flex items-start justify-between mb-2">
                              <div className="flex-1 min-w-0 mr-4">
                                  <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">ID: {sub.id}</span>
                                  {sub.original_url && sub.original_url.startsWith('http') ? (
                                      <a 
                                        href={sub.original_url} 
                                        target="_blank" 
                                        rel="noreferrer"
                                        className="block text-lg font-medium text-cyan-400 hover:underline break-all"
                                        onClick={(e) => e.stopPropagation()}
                                      >
                                          {sub.original_url}
                                      </a>
                                  ) : (
                                      <h3 className="text-lg font-medium text-white truncate max-w-md" title={sub.original_url}>
                                          {sub.original_url || "Uploaded Local File"}
                                      </h3>
                                  )}
                              </div>
                              
                              {/* Verdict Badge */}
                              {sub.summary.isAi ? (
                                  <div className="flex items-center gap-1.5 bg-rose-500/10 text-rose-400 px-3 py-1 rounded-full text-xs font-bold border border-rose-500/20">
                                      <AlertCircle className="w-3 h-3" /> AI Generated
                                  </div>
                              ) : (
                                   <div className="flex items-center gap-1.5 bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-full text-xs font-bold border border-emerald-500/20">
                                      <CheckCircle className="w-3 h-3" /> Real Footage
                                  </div>
                              )}
                          </div>
                          
                          <div className="flex flex-wrap gap-4 text-sm text-slate-400">
                              <div className="flex items-center gap-1.5">
                                  <Calendar className="w-3 h-3" />
                                  {formatDate(sub.created_at)}
                              </div>
                              <div className="flex items-center gap-1.5 text-slate-500" title="Submitter IP">
                                  <Globe className="w-3 h-3" />
                                  {sub.ip_address || "Unknown IP"}
                              </div>
                              {sub.summary.modelDetected && (
                                   <div>
                                       Generator: <span className="text-cyan-500">{sub.summary.modelDetected}</span>
                                   </div>
                              )}
                               {sub.summary.confidence && (
                                   <div>
                                       Confidence: <span className="text-white">{sub.summary.confidence}%</span>
                                   </div>
                              )}
                          </div>
                      </div>
                  </div>
              ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-8 flex items-center justify-center gap-4">
              <button
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-800 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </button>
              
              <div className="flex items-center gap-2">
                <span className="text-slate-400 text-sm">
                  Page <span className="text-white font-medium">{currentPage}</span> of <span className="text-white font-medium">{totalPages}</span>
                </span>
              </div>

              <button
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-800 transition-colors"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default History;
