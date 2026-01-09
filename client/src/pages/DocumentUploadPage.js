import React from 'react';
import UploadForm from '../components/UploadForm';
import { uploadDocument } from '../services/documentService';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const DocumentUploadPage = () => {
  const navigate = useNavigate(); // ✅ Required for redirection

  const handleUpload = (docData) => {
    // Upload is handled inside UploadForm.js
    console.log('Document upload success:', docData);
    // Navigation is also handled by UploadForm after simulation
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#0f0c29] via-[#302b63] to-[#24243e] text-white px-4 py-10 relative overflow-hidden">
      {/* Animated Background Blobs */}
      <div className="absolute top-0 -left-20 w-[600px] h-[600px] bg-pink-500 opacity-20 rounded-full blur-3xl animate-pulse z-0" />
      <div className="absolute bottom-0 -right-20 w-[500px] h-[500px] bg-blue-500 opacity-20 rounded-full blur-2xl animate-ping z-0" />

      <motion.div
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="relative w-full max-w-2xl p-8 bg-white/10 backdrop-blur-md border border-white/20 rounded-xl shadow-xl z-10 text-white"
      >
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold mb-2">🔒 Secure Document Upload</h1>
          <p className="text-gray-300 text-sm">
            Upload your documents for AI-powered verification and analysis
          </p>
        </div>

        {/* Security Features */}
        <div className="mb-6 grid grid-cols-3 gap-4 text-center">
          <div className="bg-white/5 p-3 rounded-lg border border-white/10">
            <div className="text-2xl mb-1">🔐</div>
            <p className="text-xs text-gray-300">End-to-End Encrypted</p>
          </div>
          <div className="bg-white/5 p-3 rounded-lg border border-white/10">
            <div className="text-2xl mb-1">🤖</div>
            <p className="text-xs text-gray-300">AI-Powered Analysis</p>
          </div>
          <div className="bg-white/5 p-3 rounded-lg border border-white/10">
            <div className="text-2xl mb-1">⚡</div>
            <p className="text-xs text-gray-300">Real-time Processing</p>
          </div>
        </div>

        {/* Upload Form */}
        <div className="text-white">
          <UploadForm onUploadSuccess={handleUpload} />
        </div>
      </motion.div>
    </div>
  );
};

export default DocumentUploadPage;
