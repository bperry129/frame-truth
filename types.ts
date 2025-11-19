export interface TrajectoryPoint {
  x: number;
  y: number;
  frame: number;
}

export interface AnalysisResult {
  isAi: boolean;
  confidence: number;
  curvatureScore: number; // 0 to 100, where higher is more curved (AI)
  distanceScore: number; // 0 to 100
  reasoning: string[];
  trajectoryData: TrajectoryPoint[];
  modelDetected?: string; // e.g., "Sora", "Pika", "Runway"
}

export interface VideoMetadata {
  name: string;
  size: number;
  type: string;
  url: string;
  title?: string;
}

export type AnalysisStatus = 'idle' | 'uploading' | 'downloading' | 'processing' | 'complete' | 'error';
