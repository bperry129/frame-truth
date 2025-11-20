import { AnalysisResult } from "../types";
import { getApiUrl } from '../src/config';

// Now we delegate analysis to the secure backend
export const analyzeVideo = async (filename: string, originalUrl: string = ""): Promise<{ result: AnalysisResult; submission_id: string }> => {
  try {
    const response = await fetch(getApiUrl('analyze'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, original_url: originalUrl })
    });

    if (!response.ok) {
       if (response.status === 429) {
           throw new Error("Daily submission limit reached (5/5). Please contact support.");
       }
       const err = await response.json();
       throw new Error(err.detail || "Analysis failed");
    }

    const data = await response.json();
    // data = { submission_id, result }
    return {
        result: data.result as AnalysisResult,
        submission_id: data.submission_id
    };

  } catch (error: any) {
    console.error("Backend Analysis Error:", error);
    throw error;
  }
};
