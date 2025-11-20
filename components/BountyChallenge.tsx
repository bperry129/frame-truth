import React, { useState } from 'react';
import { Trophy, Target, DollarSign, X, Mail, CheckCircle, AlertTriangle, Zap } from 'lucide-react';

interface BountyChallengeProps {
  isOpen: boolean;
  onClose: () => void;
}

const BountyChallenge: React.FC<BountyChallengeProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-in fade-in duration-300">
      <div className="bg-slate-900 border border-orange-500/30 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="relative bg-gradient-to-r from-orange-600 to-red-600 p-6 rounded-t-2xl">
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 w-8 h-8 bg-black/20 hover:bg-black/40 rounded-full flex items-center justify-center transition-colors"
          >
            <X className="w-4 h-4 text-white" />
          </button>
          
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 bg-yellow-400 rounded-full flex items-center justify-center shadow-lg">
              <Trophy className="w-8 h-8 text-yellow-900" />
            </div>
            <div>
              <h2 className="text-3xl font-bold text-white mb-1">BREAK OUR AI VIDEO DETECTOR</h2>
              <p className="text-orange-100 text-lg font-medium">WIN $25!</p>
            </div>
          </div>
          
          <p className="text-orange-100 text-lg">
            Think you can fool our system? <strong>Prove it.</strong>
          </p>
        </div>

        <div className="p-6 space-y-8">
          {/* Challenge Description */}
          <div className="bg-slate-950/50 rounded-xl p-6 border border-slate-800">
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 bg-red-500/10 rounded-full flex items-center justify-center flex-shrink-0">
                <Zap className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white mb-2">The Challenge</h3>
                <p className="text-slate-300 leading-relaxed">
                  Our AI Video Detector is scary accurate… but if <strong className="text-orange-400">YOU</strong> can break it, 
                  we'll pay you <strong className="text-green-400">$25</strong>. Yes, really. Cash. Via PayPal.
                </p>
              </div>
            </div>
            
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <h4 className="text-orange-400 font-semibold mb-2">Your mission:</h4>
              <ul className="text-slate-300 space-y-1 text-sm">
                <li>• Upload a video</li>
                <li>• If our system marks it <strong className="text-red-400">REAL</strong> when it's 100% AI, or marks it <strong className="text-blue-400">AI</strong> when it's 100% REAL, you win the bounty</li>
              </ul>
              <p className="text-slate-400 mt-3 text-sm">
                Easy. Or is it? <span className="text-red-400">😈</span>
              </p>
            </div>
          </div>

          {/* How to Win */}
          <div className="bg-gradient-to-br from-green-900/20 to-emerald-900/20 rounded-xl p-6 border border-green-500/20">
            <div className="flex items-center gap-3 mb-4">
              <Target className="w-6 h-6 text-green-400" />
              <h3 className="text-xl font-bold text-white">🧩 How to Win $25</h3>
            </div>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 text-white text-sm font-bold">1</div>
                  <p className="text-slate-300">Upload a video on our site</p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 text-white text-sm font-bold">2</div>
                  <p className="text-slate-300">If the detection result is obviously wrong, save your Submission ID</p>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 text-white text-sm font-bold">3</div>
                  <p className="text-slate-300">Email it to us at <a href="mailto:admin@frametruth.com" className="text-cyan-400 hover:underline">admin@frametruth.com</a></p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 text-white text-sm font-bold">4</div>
                  <p className="text-slate-300">We review it, verify it, and if the video qualifies — you get $25</p>
                </div>
              </div>
            </div>
          </div>

          {/* Valid Submissions */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-blue-900/20 rounded-xl p-6 border border-blue-500/20">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="w-6 h-6 text-blue-400" />
                <h3 className="text-lg font-bold text-white">🎯 What Counts as Valid</h3>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h4 className="text-blue-400 font-semibold mb-2">✔ 100% REAL video</h4>
                  <ul className="text-slate-300 text-sm space-y-1">
                    <li>• Recorded with a camera</li>
                    <li>• Not AI-enhanced, not AI-upscaled, not AI-edited</li>
                    <li>• Just pure real-world footage</li>
                  </ul>
                </div>
                
                <div className="border-t border-slate-700 pt-4">
                  <h4 className="text-blue-400 font-semibold mb-2">✔ 100% AI-generated video</h4>
                  <p className="text-slate-300 text-sm mb-2">Created fully with:</p>
                  <div className="grid grid-cols-2 gap-1 text-xs text-slate-400">
                    <span>• Sora</span>
                    <span>• Runway</span>
                    <span>• Pika</span>
                    <span>• HeyGen</span>
                    <span>• Synthesia</span>
                    <span>• Stable Video</span>
                    <span>• Anything similar</span>
                  </div>
                  <p className="text-slate-500 text-xs mt-2">
                    <strong>But:</strong> No cartoon / animation / video game / CGI content (We're detecting AI realism, not Pixar.)
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-red-900/20 rounded-xl p-6 border border-red-500/20">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-6 h-6 text-red-400" />
                <h3 className="text-lg font-bold text-white">❌ What Does NOT Count</h3>
              </div>
              
              <p className="text-slate-300 text-sm mb-4">We want real attempts — not exploits.</p>
              
              <ul className="text-slate-300 text-sm space-y-2">
                <li>• No mixed videos (half AI, half real)</li>
                <li>• No filters that add AI artifacts</li>
                <li>• No deepfakes of real videos</li>
                <li>• No TikTok filters labeled "AI" but actually pulling from real footage</li>
                <li>• No cartoon, anime, video-game, or CGI renders</li>
                <li>• No screen-recording of AI models training</li>
                <li>• No attempts to manipulate or hack our backend</li>
                <li>• No attempts to force weird compression glitches just to confuse the system</li>
              </ul>
              
              <div className="bg-red-500/10 rounded-lg p-3 mt-4 border border-red-500/20">
                <p className="text-red-300 text-sm font-medium">
                  This is a fairness challenge — not a hacking challenge.
                </p>
              </div>
            </div>
          </div>

          {/* Verification Requirements */}
          <div className="bg-slate-950/50 rounded-xl p-6 border border-slate-800">
            <div className="flex items-center gap-3 mb-4">
              <Mail className="w-6 h-6 text-purple-400" />
              <h3 className="text-xl font-bold text-white">🔍 Verification Requirements</h3>
            </div>
            
            <p className="text-slate-300 mb-4">To claim the bounty, you must provide:</p>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                <h4 className="text-blue-400 font-semibold mb-3">If your video is REAL:</h4>
                <ul className="text-slate-300 text-sm space-y-1">
                  <li>✔ A link to the original source</li>
                  <li>✔ Or EXIF metadata / behind-the-scenes proof</li>
                  <li>✔ Or a camera-original file</li>
                </ul>
              </div>
              
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                <h4 className="text-red-400 font-semibold mb-3">If your video is AI:</h4>
                <ul className="text-slate-300 text-sm space-y-1">
                  <li>✔ The generation platform (Sora, Runway, Pika, etc.)</li>
                  <li>✔ The prompt</li>
                  <li>✔ Or the creation metadata</li>
                  <li>✔ Or the platform's built-in watermark / log</li>
                </ul>
              </div>
            </div>
            
            <div className="bg-yellow-500/10 rounded-lg p-4 mt-4 border border-yellow-500/20">
              <p className="text-yellow-200 text-sm">
                <strong>Important:</strong> We must be able to prove beyond doubt that the video is 100% real or 100% AI. 
                If confirmation isn't possible → the submission cannot win.
              </p>
            </div>
          </div>

          {/* CTA */}
          <div className="text-center bg-gradient-to-r from-orange-600/20 to-red-600/20 rounded-xl p-6 border border-orange-500/30">
            <div className="flex items-center justify-center gap-3 mb-4">
              <DollarSign className="w-8 h-8 text-green-400" />
              <h3 className="text-2xl font-bold text-white">Ready to Win $25?</h3>
            </div>
            <p className="text-slate-300 mb-6">
              Close this modal and upload your video to get started. Good luck! 🍀
            </p>
            <button 
              onClick={onClose}
              className="bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 text-white font-bold py-3 px-8 rounded-lg transition-all shadow-lg shadow-orange-900/20 active:scale-95"
            >
              Start Challenge
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Floating Button Component
export const BountyFloatingButton: React.FC<{ onClick: () => void }> = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="fixed bottom-6 right-6 z-40 group"
      aria-label="Bounty Challenge"
    >
      <div className="relative">
        {/* Glow effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-red-500 rounded-full blur-lg opacity-60 group-hover:opacity-80 transition-opacity animate-pulse"></div>
        
        {/* Main button */}
        <div className="relative w-16 h-16 bg-gradient-to-r from-orange-600 to-red-600 rounded-full flex items-center justify-center shadow-2xl group-hover:scale-110 transition-transform cursor-pointer border-2 border-orange-400/50">
          <Trophy className="w-8 h-8 text-white drop-shadow-lg" />
        </div>
        
        {/* Bounty badge */}
        <div className="absolute -top-2 -right-2 bg-green-500 text-white text-xs font-bold px-2 py-1 rounded-full shadow-lg border-2 border-white">
          $25
        </div>
        
        {/* Tooltip */}
        <div className="absolute bottom-full right-0 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
          <div className="bg-slate-900 text-white text-sm font-medium px-3 py-2 rounded-lg shadow-xl border border-slate-700 whitespace-nowrap">
            Break Our AI - Win $25!
            <div className="absolute top-full right-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-900"></div>
          </div>
        </div>
      </div>
    </button>
  );
};

export default BountyChallenge;
