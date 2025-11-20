import { VideoMetadata } from '../types';
import { getApiUrl, API_BASE_URL } from '../src/config';

/**
 * HYBRID DOWNLOADER SERVICE
 * 
 * Priority 1: Local Python Backend (http://localhost:8000)
 * Priority 2: Public APIs (Cobalt) as fallback
 */

interface DownloaderOptions {
  onProgress: (stage: string) => void;
}

// --- Priority 1: Backend (Local or Serverless) ---

const downloadFromBackend = async (url: string, report: (msg: string) => void): Promise<{ blob: Blob; filename: string; meta?: any } | null> => {
  try {
    report("Starting video extraction...");
    
    const response = await fetch(getApiUrl('download'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });

    if (!response.ok) {
      const errorText = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errorText.detail || `Server Error: ${response.status}`);
    }

    report("Download successful. Fetching video data...");
    
    // Check if response is JSON (new format) or Blob (old format)
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
        const data = await response.json();
        // data = { filename: "...", url: "..." }
        if (!data.url) throw new Error("Invalid backend response");
        
        const fileRes = await fetch(data.url);
        const blob = await fileRes.blob();
        return { blob, filename: data.filename, meta: data.meta };
    } else {
        // Fallback if backend is old version (returns file directly)
        const blob = await response.blob();
        return { blob, filename: `video_${Date.now()}.mp4` };
    }

  } catch (error: any) {
    // If fetch failed (connection refused), return null to trigger fallback
    if (error.name === 'TypeError' || error.message.includes('Failed to fetch') || error.name === 'AbortError') {
        return null;
    }
    throw error; // Rethrow logic errors (e.g., 400 Bad Request from backend)
  }
};

// --- Priority 2: Public API Fallback (Cobalt) ---

const downloadFromPublicApi = async (targetUrl: string, report: (msg: string) => void): Promise<Blob> => {
    report("Local backend unavailable. Attempting public cloud fallback...");

    const instances = [
        "https://cobalt.pub",
        "https://api.cobalt.tools", 
        "https://api.wuk.sh",
    ];

    const shuffled = instances.sort(() => 0.5 - Math.random());

    for (const instance of shuffled) {
        try {
            report(`Trying Cloud Engine (${new URL(instance).hostname})...`);
            
            const response = await fetch(`${instance}/api/json`, {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: targetUrl,
                    filenamePattern: "basic" 
                })
            });

            if (!response.ok) continue;

            const data = await response.json();
            
            if (data.status === 'error') throw new Error(data.text);
            
            const streamUrl = data.url || (data.picker && data.picker[0]?.url);
            if (!streamUrl) continue;

            report("Downloading stream...");
            // Try via Proxy to avoid CORS
            const proxyUrl = `https://corsproxy.io/?${encodeURIComponent(streamUrl)}`;
            const proxyRes = await fetch(proxyUrl);
            if (!proxyRes.ok) throw new Error("Stream download failed");
            
            return await proxyRes.blob();

        } catch (e) {
            // Continue to next instance
        }
    }

    throw new Error("Cloud extraction failed. Please run 'python backend/server.py' locally for reliable downloading.");
};


// --- Main Export ---

export const downloadVideo = async (
  urlInput: string, 
  options?: DownloaderOptions
): Promise<{ file: File; meta: Partial<VideoMetadata>; backendFilename?: string }> => {
  
  const report = (msg: string) => options?.onProgress && options.onProgress(msg);
  let url = urlInput.trim();
  if (!url.startsWith('http')) url = `https://${url}`;

  try {
    // 1. Try Backend (Local or Serverless)
    let backendData = await downloadFromBackend(url, report);
    let blob: Blob | null = null;
    let backendFilename: string | undefined;
    let backendMeta: any = {};

    if (backendData) {
        blob = backendData.blob;
        backendFilename = backendData.filename;
        backendMeta = backendData.meta || {};
    } else {
        // 2. Fallback to Cloud
        blob = await downloadFromPublicApi(url, report);
    }

    const filename = backendFilename || `video_${Date.now()}.mp4`;
    const file = new File([blob], filename, { type: 'video/mp4' });

    // Merge backend meta if available
    const metaName = backendMeta.title || filename;

    return {
      file,
      meta: {
        ...backendMeta,
        name: metaName,
        size: file.size,
        type: file.type,
        url: URL.createObjectURL(file)
      },
      backendFilename
    };

  } catch (error: any) {
    console.error("Downloader Error:", error);
    throw new Error(error.message || "Download failed");
  }
};
