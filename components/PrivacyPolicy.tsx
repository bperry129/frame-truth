import React from 'react';
import { Shield, Eye, Database, Lock, Mail } from 'lucide-react';

const PrivacyPolicy: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="text-center mb-12">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Shield className="w-8 h-8 text-cyan-400" />
          <h1 className="text-4xl font-bold text-white">Privacy Policy</h1>
        </div>
        <p className="text-slate-400 text-lg">
          Your privacy is important to us. This policy explains how we collect, use, and protect your information.
        </p>
        <p className="text-slate-500 text-sm mt-2">
          Last updated: November 19, 2025
        </p>
      </div>

      <div className="space-y-8">
        {/* Information We Collect */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Database className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Information We Collect</h2>
          </div>
          <div className="space-y-4 text-slate-300">
            <div>
              <h3 className="text-lg font-medium text-white mb-2">Video Content</h3>
              <p>When you upload or submit video URLs for analysis, we temporarily process the video content to perform AI detection analysis. Videos are not permanently stored unless required for service improvement.</p>
            </div>
            <div>
              <h3 className="text-lg font-medium text-white mb-2">Analysis Results</h3>
              <p>We store analysis results and submission metadata to provide you with shareable links and to improve our detection algorithms.</p>
            </div>
            <div>
              <h3 className="text-lg font-medium text-white mb-2">Technical Information</h3>
              <p>We collect IP addresses for rate limiting and abuse prevention. No personal identifying information is collected unless you voluntarily provide it through feedback forms.</p>
            </div>
          </div>
        </section>

        {/* How We Use Information */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Eye className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">How We Use Your Information</h2>
          </div>
          <div className="space-y-3 text-slate-300">
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
              <p>Provide AI video detection and analysis services</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
              <p>Improve our detection algorithms and service quality</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
              <p>Prevent abuse and maintain service availability</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
              <p>Respond to user feedback and support requests</p>
            </div>
          </div>
        </section>

        {/* Data Security */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Lock className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Data Security</h2>
          </div>
          <div className="space-y-4 text-slate-300">
            <p>We implement appropriate technical and organizational measures to protect your information against unauthorized access, alteration, disclosure, or destruction.</p>
            <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-4">
              <h3 className="text-white font-medium mb-2">Security Measures Include:</h3>
              <ul className="space-y-1 text-sm">
                <li>• Encrypted data transmission (HTTPS)</li>
                <li>• Secure server infrastructure</li>
                <li>• Regular security audits and updates</li>
                <li>• Limited data retention periods</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Data Retention */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Database className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Data Retention</h2>
          </div>
          <div className="space-y-3 text-slate-300">
            <p><strong className="text-white">Video Files:</strong> Temporarily processed and deleted after analysis completion</p>
            <p><strong className="text-white">Analysis Results:</strong> Stored indefinitely to provide shareable links and service improvement</p>
            <p><strong className="text-white">IP Addresses:</strong> Retained for 30 days for rate limiting purposes</p>
            <p><strong className="text-white">Feedback:</strong> Retained until resolved or for service improvement purposes</p>
          </div>
        </section>

        {/* Third-Party Services */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Eye className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Third-Party Services</h2>
          </div>
          <div className="space-y-3 text-slate-300">
            <p>Our service uses Google's Gemini AI for semantic analysis of video content. Video data sent to Gemini is processed according to Google's privacy policies and is not stored by Google for model training.</p>
            <p>We may use third-party services for video downloading from public platforms (YouTube, Instagram, etc.) which operate under their respective privacy policies.</p>
          </div>
        </section>

        {/* Your Rights */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Your Rights</h2>
          </div>
          <div className="space-y-3 text-slate-300">
            <p>You have the right to:</p>
            <div className="grid md:grid-cols-2 gap-3 mt-4">
              <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3">
                <p className="text-white font-medium">Access</p>
                <p className="text-sm">Request information about your data</p>
              </div>
              <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3">
                <p className="text-white font-medium">Deletion</p>
                <p className="text-sm">Request deletion of your submissions</p>
              </div>
              <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3">
                <p className="text-white font-medium">Correction</p>
                <p className="text-sm">Request correction of inaccurate data</p>
              </div>
              <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3">
                <p className="text-white font-medium">Portability</p>
                <p className="text-sm">Request export of your data</p>
              </div>
            </div>
          </div>
        </section>

        {/* Contact Information */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Mail className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Contact Us</h2>
          </div>
          <div className="text-slate-300">
            <p className="mb-4">If you have any questions about this Privacy Policy or wish to exercise your rights, please contact us:</p>
            <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-4">
              <p className="text-white font-medium mb-2">FrameTruth Privacy Team</p>
              <a href="mailto:admin@frametruth.com" className="text-cyan-400 hover:text-cyan-300 transition-colors">
                admin@frametruth.com
              </a>
            </div>
          </div>
        </section>

        {/* Updates */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <h2 className="text-2xl font-semibold text-white mb-4">Policy Updates</h2>
          <div className="text-slate-300">
            <p>We may update this Privacy Policy from time to time. We will notify users of any material changes by posting the new Privacy Policy on this page and updating the "Last updated" date.</p>
            <p className="mt-3 text-slate-400 text-sm">
              Continued use of our service after any changes constitutes acceptance of the updated Privacy Policy.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
};

export default PrivacyPolicy;
