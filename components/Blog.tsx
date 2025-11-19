import React from 'react';
import { ArrowRight, UserCheck, Shield, Eye, Cpu } from 'lucide-react';

const articles = [
  {
    title: "How to Tell if a Video is AI Generated: Top 5 Signs",
    desc: "Learn the common artifacts of AI video generation, from texture boiling to object permanence failures. A comprehensive guide to detecting deepfakes manually.",
    category: "Guide",
    icon: <Eye className="w-5 h-5 text-cyan-400" />
  },
  {
    title: "Is This Video Real or Fake? The Truth About Viral Deepfakes",
    desc: "We analyze viral TikTok and YouTube deepfakes to show you exactly how to verify video authenticity using forensic tools.",
    category: "Case Study",
    icon: <Shield className="w-5 h-5 text-purple-400" />
  },
  {
    title: "Best AI Video Detectors of 2025: Authenticity Checker Review",
    desc: "Comparing the top media verification tools. Why FrameTruth's latent trajectory analysis outperforms standard deepfake detection APIs.",
    category: "Review",
    icon: <Cpu className="w-5 h-5 text-emerald-400" />
  },
  {
    title: "Detect Fake TikTok Videos: A Parent's Guide to AI Content",
    desc: "Protect your family from misinformation. Simple steps to verify if a TikTok or Instagram Reel is AI-generated or authentic footage.",
    category: "Safety",
    icon: <UserCheck className="w-5 h-5 text-rose-400" />
  },
  {
    title: "Video Forensics 101: Understanding Synthetic Video Detection",
    desc: "A technical dive into how video integrity analysis works. From metadata inspection to advanced semantic consistency checks.",
    category: "Technical",
    icon: <Shield className="w-5 h-5 text-blue-400" />
  },
  {
    title: "Did AI Make This Video? Understanding Sora and Runway Gen-3",
    desc: "Generative AI models are getting better. Here is how to spot the subtle flaws in videos created by the latest text-to-video engines.",
    category: "Technology",
    icon: <Cpu className="w-5 h-5 text-orange-400" />
  },
  {
    title: "Deepfake Detection API for Enterprise Media Verification",
    desc: "How news organizations and social platforms use automated video authenticity software to filter out synthetic propaganda.",
    category: "Enterprise",
    icon: <Shield className="w-5 h-5 text-indigo-400" />
  },
  {
    title: "Real or AI Video Checker: Why Metadata Isn't Enough",
    desc: "Why simple metadata extraction fails to detect modern deepfakes, and why visual forensic analysis is the only reliable method.",
    category: "Analysis",
    icon: <Eye className="w-5 h-5 text-cyan-400" />
  },
  {
    title: "How to Verify Video Authenticity Online for Free",
    desc: "A step-by-step tutorial on using online tools to check if a news clip or viral video is legitimate or specific AI manipulaton.",
    category: "Tutorial",
    icon: <UserCheck className="w-5 h-5 text-emerald-400" />
  },
  {
    title: "The Future of Video Integrity Analysis",
    desc: "As AI evolves, so must detection. Exploring the next generation of synthetic video detectors and digital watermarking.",
    category: "Industry",
    icon: <Cpu className="w-5 h-5 text-purple-400" />
  }
];

const Blog: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto px-4 py-12 animate-fade-in">
      <div className="text-center mb-16">
        <h2 className="text-4xl font-bold text-white mb-4 tracking-tight">Deepfake Detection <span className="text-cyan-400">Insights</span></h2>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto">
          Expert guides, technical analysis, and tutorials on how to <strong>detect AI videos</strong> and verify media authenticity in the age of generative AI.
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {articles.map((article, idx) => (
          <article key={idx} className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 hover:border-cyan-500/30 transition-all hover:bg-slate-800/50 group cursor-pointer">
            <div className="flex items-center gap-2 mb-4">
              <span className="px-2 py-1 rounded text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700">
                {article.category}
              </span>
              {article.icon}
            </div>
            <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-cyan-400 transition-colors">
              {article.title}
            </h3>
            <p className="text-slate-400 text-sm leading-relaxed mb-4">
              {article.desc}
            </p>
            <div className="flex items-center text-cyan-500 text-sm font-medium gap-1 opacity-0 group-hover:opacity-100 transition-opacity transform translate-x-[-10px] group-hover:translate-x-0">
              Read Article <ArrowRight className="w-4 h-4" />
            </div>
          </article>
        ))}
      </div>

      {/* SEO Footer Text */}
      <div className="mt-20 pt-12 border-t border-slate-900 text-slate-500 text-sm leading-relaxed">
        <h3 className="text-slate-300 font-semibold mb-4 text-base">Why Use an AI Video Detector?</h3>
        <p className="mb-4">
          In today's digital landscape, distinguishing between <strong>real or fake videos</strong> is crucial. 
          Advanced generative models like Sora, Runway, and Pika can create hyper-realistic footage that deceives the human eye. 
          Our <strong>AI video authenticity checker</strong> provides a reliable solution for content creators, journalists, and researchers to <strong>verify video authenticity</strong>.
        </p>
        <p>
          Whether you need to <strong>detect fake TikTok videos</strong>, verify news footage, or perform <strong>video integrity analysis</strong> for enterprise purposes, 
          FrameTruth offers state-of-the-art forensic tools. By analyzing the latent space trajectory of video frames, we can scientifically determine: <em>"Is this video AI-generated?"</em>
        </p>
      </div>
    </div>
  );
};

export default Blog;
