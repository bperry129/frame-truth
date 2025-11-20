import React from 'react';
import { FileText, AlertTriangle, Scale, Shield, Mail } from 'lucide-react';

const TermsOfService: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="text-center mb-12">
        <div className="flex items-center justify-center gap-3 mb-4">
          <FileText className="w-8 h-8 text-cyan-400" />
          <h1 className="text-4xl font-bold text-white">Terms of Service</h1>
        </div>
        <p className="text-slate-400 text-lg">
          Please read these terms carefully before using our AI video detection service.
        </p>
        <p className="text-slate-500 text-sm mt-2">
          Last updated: November 19, 2025
        </p>
      </div>

      <div className="space-y-8">
        {/* Acceptance of Terms */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Scale className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Acceptance of Terms</h2>
          </div>
          <div className="text-slate-300 space-y-3">
            <p>By accessing and using FrameTruth's AI video detection service, you accept and agree to be bound by the terms and provision of this agreement.</p>
            <p>If you do not agree to abide by the above, please do not use this service.</p>
          </div>
        </section>

        {/* Service Description */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Service Description</h2>
          </div>
          <div className="text-slate-300 space-y-3">
            <p>FrameTruth provides AI-powered video authenticity detection services. Our service analyzes video content to determine the likelihood that it was generated or manipulated using artificial intelligence.</p>
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-amber-200 font-medium mb-2">Important Disclaimers:</p>
                  <ul className="text-amber-100/90 space-y-1 text-sm">
                    <li>• Results are probabilistic and not 100% accurate</li>
                    <li>• Service is provided for informational purposes only</li>
                    <li>• Not suitable for legal or forensic evidence</li>
                    <li>• May produce false positives or false negatives</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Acceptable Use */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Acceptable Use</h2>
          </div>
          <div className="text-slate-300 space-y-4">
            <div>
              <h3 className="text-white font-medium mb-2">You may use our service to:</h3>
              <ul className="space-y-1 text-sm">
                <li>• Analyze videos you own or have permission to analyze</li>
                <li>• Verify authenticity of publicly available content</li>
                <li>• Research and educational purposes</li>
                <li>• Personal verification of suspicious content</li>
              </ul>
            </div>
            <div>
              <h3 className="text-white font-medium mb-2">You may NOT use our service to:</h3>
              <ul className="space-y-1 text-sm text-rose-300">
                <li>• Analyze copyrighted content without permission</li>
                <li>• Harass, defame, or harm others</li>
                <li>• Violate any applicable laws or regulations</li>
                <li>• Attempt to reverse engineer our algorithms</li>
                <li>• Overwhelm our servers with excessive requests</li>
                <li>• Upload malicious or harmful content</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Rate Limits and Beta Status */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Beta Service & Limitations</h2>
          </div>
          <div className="text-slate-300 space-y-3">
            <p>FrameTruth is currently in beta. The service is provided "as is" with the following limitations:</p>
            <div className="grid md:grid-cols-2 gap-4 mt-4">
              <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-4">
                <h3 className="text-white font-medium mb-2">Rate Limits</h3>
                <p className="text-sm">5 submissions per day per IP address</p>
              </div>
              <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-4">
                <h3 className="text-white font-medium mb-2">File Size</h3>
                <p className="text-sm">Maximum 50MB per video file</p>
              </div>
              <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-4">
                <h3 className="text-white font-medium mb-2">Availability</h3>
                <p className="text-sm">No uptime guarantees during beta</p>
              </div>
              <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-4">
                <h3 className="text-white font-medium mb-2">Support</h3>
                <p className="text-sm">Best effort support via email</p>
              </div>
            </div>
          </div>
        </section>

        {/* Intellectual Property */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Scale className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Intellectual Property</h2>
          </div>
          <div className="text-slate-300 space-y-3">
            <p>The FrameTruth service, including its algorithms, interface, and documentation, is protected by intellectual property laws.</p>
            <div className="space-y-2">
              <p><strong className="text-white">Your Content:</strong> You retain ownership of videos you upload. By using our service, you grant us a limited license to process your content for analysis purposes.</p>
              <p><strong className="text-white">Our Service:</strong> All rights to our detection algorithms, interface, and analysis methods remain with FrameTruth.</p>
              <p><strong className="text-white">Results:</strong> Analysis results are provided for your use, but our underlying methods remain proprietary.</p>
            </div>
          </div>
        </section>

        {/* Disclaimers and Limitations */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Disclaimers and Limitations</h2>
          </div>
          <div className="text-slate-300 space-y-4">
            <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg p-4">
              <h3 className="text-rose-200 font-medium mb-2">No Warranty</h3>
              <p className="text-rose-100/90 text-sm">
                The service is provided "as is" without any warranties, express or implied. We do not guarantee the accuracy, reliability, or availability of our detection results.
              </p>
            </div>
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
              <h3 className="text-amber-200 font-medium mb-2">Limitation of Liability</h3>
              <p className="text-amber-100/90 text-sm">
                FrameTruth shall not be liable for any direct, indirect, incidental, special, or consequential damages resulting from the use or inability to use our service.
              </p>
            </div>
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
              <h3 className="text-blue-200 font-medium mb-2">Accuracy Disclaimer</h3>
              <p className="text-blue-100/90 text-sm">
                AI detection is not perfect. Results should be considered as one factor among many when evaluating video authenticity. Do not rely solely on our results for important decisions.
              </p>
            </div>
          </div>
        </section>

        {/* Privacy and Data */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Privacy and Data Handling</h2>
          </div>
          <div className="text-slate-300 space-y-3">
            <p>Your privacy is important to us. Please review our Privacy Policy for detailed information about how we collect, use, and protect your data.</p>
            <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-4">
              <h3 className="text-white font-medium mb-2">Key Points:</h3>
              <ul className="space-y-1 text-sm">
                <li>• Videos are processed temporarily and not permanently stored</li>
                <li>• Analysis results may be retained for service improvement</li>
                <li>• We implement security measures to protect your data</li>
                <li>• Third-party AI services may process your content</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Termination */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Termination</h2>
          </div>
          <div className="text-slate-300 space-y-3">
            <p>We reserve the right to terminate or suspend access to our service immediately, without prior notice, for any reason, including but not limited to:</p>
            <ul className="space-y-1 text-sm ml-4">
              <li>• Violation of these Terms of Service</li>
              <li>• Abuse of our service or systems</li>
              <li>• Illegal or harmful activities</li>
              <li>• Technical or business reasons</li>
            </ul>
            <p>You may discontinue use of our service at any time.</p>
          </div>
        </section>

        {/* Changes to Terms */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <FileText className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Changes to Terms</h2>
          </div>
          <div className="text-slate-300 space-y-3">
            <p>We reserve the right to modify these terms at any time. Changes will be effective immediately upon posting to this page.</p>
            <p>Your continued use of the service after any changes constitutes acceptance of the new terms.</p>
            <p className="text-slate-400 text-sm">
              We recommend reviewing these terms periodically for any updates.
            </p>
          </div>
        </section>

        {/* Contact Information */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Mail className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Contact Information</h2>
          </div>
          <div className="text-slate-300">
            <p className="mb-4">If you have any questions about these Terms of Service, please contact us:</p>
            <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-4">
              <p className="text-white font-medium mb-2">FrameTruth Legal Team</p>
              <a href="mailto:admin@frametruth.com" className="text-cyan-400 hover:text-cyan-300 transition-colors">
                admin@frametruth.com
              </a>
            </div>
          </div>
        </section>

        {/* Governing Law */}
        <section className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Scale className="w-6 h-6 text-cyan-400" />
            <h2 className="text-2xl font-semibold text-white">Governing Law</h2>
          </div>
          <div className="text-slate-300">
            <p>These terms shall be governed by and construed in accordance with applicable laws. Any disputes arising from these terms or the use of our service shall be resolved through appropriate legal channels.</p>
          </div>
        </section>
      </div>
    </div>
  );
};

export default TermsOfService;
