// API Configuration for different environments
export const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api'  // In production on Vercel, API routes are at /api/*
  : 'http://localhost:8000';  // In development, use local backend

export const getApiUrl = (endpoint: string): string => {
  // Remove leading slash if present to avoid double slashes
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
  
  if (process.env.NODE_ENV === 'production') {
    return `/api/${cleanEndpoint}`;
  } else {
    return `${API_BASE_URL}/${cleanEndpoint}`;
  }
};
