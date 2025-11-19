import React from 'react';
import { Activity, Brain, Layers, Eye, FileVideo, Network } from 'lucide-react';

const Methodology: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 py-12 animate-fade-in text-slate-300">
      <div className="mb-12 text-center">
        <h2 className="text-4xl font-bold text-white mb-4 tracking-tight">The Science of <span className="text-cyan-400">Deepfake Detection</span></h2>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto">
          Our <strong>video forensics</strong> system leverages advanced principles based on the <em className="text-emerald-400">"Perceptual Straightening"</em> hypothesis to distinguish between natural footage and generative AI.
        </p>
      </div>

      <div className="space-y-12">
        {/* Core Concept */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 backdrop-blur-sm">
          <div className="flex items-start gap-4 mb-6">
            <div className="bg-cyan-500/10 p-3 rounded-lg border border-cyan-500/20">
              <Activity className="w-8 h-8 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-white mb-2">The FrameTruth Hypothesis</h3>
              <p className="text-sm font-mono text-slate-500 uppercase tracking-widest">Representation Straightening for Video</p>
            </div>
          </div>
          <p className="leading-relaxed mb-6">
            The fundamental principle behind our detection engine is that <strong>real-world dynamics are continuous and predictable</strong>. When a video camera moves through a scene, the change in the visual field follows strict physical laws. 
          </p>
          <p className="leading-relaxed">
            In high-dimensional latent space (a mathematical representation of visual content), the trajectory of a <strong>Real Video</strong> tends to be "straight" because the transition between frames is governed by consistent momentum and optics.
          </p>
          <p className="leading-relaxed mt-4">
            In contrast, <strong>AI-Generated Videos</strong> (created by Diffusion or Transformer models) are generated probabilistically. While individual frames may look photorealistic, the model often struggles to maintain global temporal consistency. This results in a "curvature" or "jitter" in the latent trajectory, as the AI effectively "hallucinates" the motion between frames rather than simulating physics.
          </p>
        </section>

        {/* Artifacts */}
        <section>
          <h3 className="text-2xl font-semibold text-white mb-8 flex items-center gap-3">
            <Eye className="w-6 h-6 text-rose-400" />
            Detectable Generative Artifacts
          </h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-700 transition-colors">
              <Layers className="w-8 h-8 text-indigo-400 mb-4" />
              <h4 className="text-lg font-medium text-white mb-2">Texture Boiling</h4>
              <p className="text-sm leading-relaxed">
                Diffusion models often struggle with high-frequency details like water, fire, or foliage. These textures "boil" or morph randomly between frames rather than flowing naturally.
              </p>
            </div>
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-700 transition-colors">
              <Brain className="w-8 h-8 text-emerald-400 mb-4" />
              <h4 className="text-lg font-medium text-white mb-2">Object Permanence</h4>
              <p className="text-sm leading-relaxed">
                AI models lack a true understanding of 3D space. Objects may vanish when obscured, merge into the background, or change shape (morphing) instead of rotating.
              </p>
            </div>
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl hover:border-slate-700 transition-colors">
              <FileVideo className="w-8 h-8 text-orange-400 mb-4" />
              <h4 className="text-lg font-medium text-white mb-2">Temporal aliasing</h4>
              <p className="text-sm leading-relaxed">
                Incoherence in lighting shadows or reflections that don't match the light source movement, or "sliding" feet that don't lock to the ground during walking.
              </p>
            </div>
          </div>
        </section>

        {/* Pipeline */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 backdrop-blur-sm">
           <div className="flex items-start gap-4 mb-6">
            <div className="bg-purple-500/10 p-3 rounded-lg border border-purple-500/20">
              <Network className="w-8 h-8 text-purple-400" />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-white mb-2">Analysis Pipeline</h3>
              <p className="text-sm font-mono text-slate-500 uppercase tracking-widest">How It Works</p>
            </div>
          </div>
          <div className="space-y-6 relative before:absolute before:left-4 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
            <div className="relative pl-12">
              <div className="absolute left-2 top-1.5 w-4 h-4 rounded-full bg-slate-800 border-2 border-purple-500"></div>
              <h4 className="text-white font-medium text-lg">1. Frame Extraction</h4>
              <p className="text-sm mt-1">
                We sample the video at equidistant temporal intervals to capture the flow of motion.
              </p>
            </div>
            <div className="relative pl-12">
              <div className="absolute left-2 top-1.5 w-4 h-4 rounded-full bg-slate-800 border-2 border-purple-500"></div>
              <h4 className="text-white font-medium text-lg">2. Visual Enrichment</h4>
              <p className="text-sm mt-1">
                Frames are processed by a Multimodal Large Language Model (Gemini 2.0 Flash) which acts as a semantic encoder, identifying objects, textures, and physical interactions.
              </p>
            </div>
            <div className="relative pl-12">
              <div className="absolute left-2 top-1.5 w-4 h-4 rounded-full bg-slate-800 border-2 border-purple-500"></div>
              <h4 className="text-white font-medium text-lg">3. Trajectory Analysis</h4>
              <p className="text-sm mt-1">
                The system simulates the latent path of the video. It calculates a <strong>Curvature Score</strong> based on the semantic stability of the scene.
              </p>
            </div>
            <div className="relative pl-12">
              <div className="absolute left-2 top-1.5 w-4 h-4 rounded-full bg-slate-800 border-2 border-purple-500"></div>
              <h4 className="text-white font-medium text-lg">4. Verdict Generation</h4>
              <p className="text-sm mt-1">
                A final probability score is calculated. If the trajectory is highly curved or physics violations are detected, the video is flagged as Synthetic.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Methodology;
