import React, { useState } from 'react';
import { MessageSquare, Send, CheckCircle, AlertCircle, Mail, User, FileText } from 'lucide-react';

const SubmitFeedback: React.FC = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    category: 'general',
    subject: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus('idle');
    setErrorMessage('');

    try {
      // Create mailto link with form data
      const subject = encodeURIComponent(`[${formData.category.toUpperCase()}] ${formData.subject}`);
      const body = encodeURIComponent(
        `Name: ${formData.name}\n` +
        `Email: ${formData.email}\n` +
        `Category: ${formData.category}\n\n` +
        `Message:\n${formData.message}\n\n` +
        `---\n` +
        `Submitted via FrameTruth Feedback Form\n` +
        `Timestamp: ${new Date().toISOString()}`
      );
      
      const mailtoLink = `mailto:admin@frametruth.com?subject=${subject}&body=${body}`;
      
      // Open email client
      window.location.href = mailtoLink;
      
      // Show success message
      setSubmitStatus('success');
      
      // Reset form after a delay
      setTimeout(() => {
        setFormData({
          name: '',
          email: '',
          category: 'general',
          subject: '',
          message: ''
        });
        setSubmitStatus('idle');
      }, 3000);
      
    } catch (error) {
      setSubmitStatus('error');
      setErrorMessage('Failed to open email client. Please email us directly at admin@frametruth.com');
    } finally {
      setIsSubmitting(false);
    }
  };

  const categories = [
    { value: 'general', label: 'General Feedback' },
    { value: 'bug', label: 'Bug Report' },
    { value: 'feature', label: 'Feature Request' },
    { value: 'accuracy', label: 'Detection Accuracy' },
    { value: 'performance', label: 'Performance Issue' },
    { value: 'support', label: 'Technical Support' },
    { value: 'business', label: 'Business Inquiry' }
  ];

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="text-center mb-12">
        <div className="flex items-center justify-center gap-3 mb-4">
          <MessageSquare className="w-8 h-8 text-cyan-400" />
          <h1 className="text-4xl font-bold text-white">Submit Feedback</h1>
        </div>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          Help us improve FrameTruth! Share your feedback, report bugs, or suggest new features.
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Feedback Form */}
        <div className="lg:col-span-2">
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
            <h2 className="text-2xl font-semibold text-white mb-6">Send us your feedback</h2>
            
            {submitStatus === 'success' && (
              <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-4 mb-6 flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-emerald-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-emerald-200 font-medium">Email client opened successfully!</p>
                  <p className="text-emerald-100/90 text-sm mt-1">
                    Please send the email from your email client. If it didn't open automatically, 
                    copy the information and email us at admin@frametruth.com
                  </p>
                </div>
              </div>
            )}

            {submitStatus === 'error' && (
              <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg p-4 mb-6 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-rose-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-rose-200 font-medium">Unable to open email client</p>
                  <p className="text-rose-100/90 text-sm mt-1">{errorMessage}</p>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Name and Email */}
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-slate-300 mb-2">
                    <User className="w-4 h-4 inline mr-2" />
                    Name
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    required
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all placeholder:text-slate-600"
                    placeholder="Your name"
                  />
                </div>
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-2">
                    <Mail className="w-4 h-4 inline mr-2" />
                    Email
                  </label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    required
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all placeholder:text-slate-600"
                    placeholder="your.email@example.com"
                  />
                </div>
              </div>

              {/* Category */}
              <div>
                <label htmlFor="category" className="block text-sm font-medium text-slate-300 mb-2">
                  <FileText className="w-4 h-4 inline mr-2" />
                  Category
                </label>
                <select
                  id="category"
                  name="category"
                  value={formData.category}
                  onChange={handleInputChange}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all"
                >
                  {categories.map(cat => (
                    <option key={cat.value} value={cat.value}>{cat.label}</option>
                  ))}
                </select>
              </div>

              {/* Subject */}
              <div>
                <label htmlFor="subject" className="block text-sm font-medium text-slate-300 mb-2">
                  Subject
                </label>
                <input
                  type="text"
                  id="subject"
                  name="subject"
                  value={formData.subject}
                  onChange={handleInputChange}
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all placeholder:text-slate-600"
                  placeholder="Brief description of your feedback"
                />
              </div>

              {/* Message */}
              <div>
                <label htmlFor="message" className="block text-sm font-medium text-slate-300 mb-2">
                  Message
                </label>
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleInputChange}
                  required
                  rows={6}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all placeholder:text-slate-600 resize-vertical"
                  placeholder="Please provide detailed information about your feedback, including steps to reproduce any issues..."
                />
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-all shadow-lg shadow-cyan-900/20 active:scale-95 flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Opening Email Client...
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    Send Feedback
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Contact Info */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Direct Contact</h3>
            <div className="space-y-3 text-slate-300">
              <div className="flex items-center gap-3">
                <Mail className="w-4 h-4 text-cyan-400" />
                <div>
                  <p className="text-white font-medium">Email</p>
                  <a href="mailto:admin@frametruth.com" className="text-cyan-400 hover:text-cyan-300 transition-colors text-sm">
                    admin@frametruth.com
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Response Time */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Response Time</h3>
            <div className="space-y-3 text-slate-300 text-sm">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-emerald-400 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <p className="text-white font-medium">General Feedback</p>
                  <p>1-3 business days</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-amber-400 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <p className="text-white font-medium">Bug Reports</p>
                  <p>Within 24 hours</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <p className="text-white font-medium">Business Inquiries</p>
                  <p>Same business day</p>
                </div>
              </div>
            </div>
          </div>

          {/* Tips */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Tips for Better Feedback</h3>
            <ul className="space-y-2 text-slate-300 text-sm">
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Be specific about the issue or suggestion</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Include steps to reproduce bugs</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Mention your browser and device type</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full mt-2 flex-shrink-0"></div>
                <span>Attach screenshots if relevant</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SubmitFeedback;
