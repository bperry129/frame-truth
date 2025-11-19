export const saveSubmission = async (
  filename: string,
  originalUrl: string,
  analysisResult: any
): Promise<{ submission_id: string }> => {
  try {
    const response = await fetch('http://localhost:8000/api/submit_analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename,
        original_url: originalUrl,
        analysis_result: analysisResult
      })
    });
    
    if (!response.ok) {
      throw new Error(`Submission failed: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (e) {
    console.error("Save API Error:", e);
    throw e;
  }
};

export const getSubmission = async (id: string): Promise<any> => {
  try {
    const response = await fetch(`http://localhost:8000/api/submission/${id}`);
    
    if (!response.ok) {
      throw new Error(`Fetch submission failed: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (e) {
    console.error("Get Submission API Error:", e);
    throw e;
  }
};

export const getSubmissions = async (user: string, pass: string): Promise<any[]> => {
  try {
    const response = await fetch('http://localhost:8000/api/submissions', {
        headers: { 
            'x-admin-user': user,
            'x-admin-pass': pass
        }
    });
    if (!response.ok) {
        throw new Error(response.statusText);
    }
    return await response.json();
  } catch (e) {
    console.error("Get Submissions API Error:", e);
    throw e;
  }
};

export const uploadFile = async (file: File): Promise<{ filename: string; url: string }> => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        return await response.json();
    } catch (e) {
        console.error("Upload API Error:", e);
        throw e;
    }
};
