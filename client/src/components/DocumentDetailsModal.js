import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { getDocumentOCR, verifyDocument } from '../services/documentService';

const DocumentDetailsModal = ({ document, isOpen, onClose, onDocumentUpdate }) => {
  const [ocrData, setOcrData] = useState(null);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrError, setOcrError] = useState(null);
  const [showOCRText, setShowOCRText] = useState(false);
  const [verifying, setVerifying] = useState(false);

  useEffect(() => {
    if (isOpen && document && document._id) {
      fetchOCRData();
    }
  }, [isOpen, document]);

  const fetchOCRData = async () => {
    try {
      setOcrLoading(true);
      setOcrError(null);
      const response = await getDocumentOCR(document._id);
      if (response.success) {
        setOcrData(response.data);
      } else {
        setOcrError(response.error || 'Failed to extract text');
      }
    } catch (error) {
      console.error('Error fetching OCR data:', error);
      setOcrError(error.message || 'Failed to extract text');
    } finally {
      setOcrLoading(false);
    }
  };

  const handleSelfApprove = async () => {
    if (!window.confirm('Are you sure you want to approve this document?')) {
      return;
    }

    try {
      setVerifying(true);
      const response = await verifyDocument(document._id, {
        isValid: true,
        notes: 'Self-approved'
      });

      if (response.success) {
        onClose();
        alert('Document approved successfully!');
        window.location.reload();
      } else {
        alert(response.message || 'Failed to approve');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to approve document.');
    } finally {
      setVerifying(false);
    }
  };

  if (!isOpen || !document) return null;

  const getStatusColor = (status) => {
    switch (status) {
      case 'verified': return 'text-green-400';
      case 'failed': case 'rejected': return 'text-red-400';
      case 'processing': case 'uploaded': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIconText = (status) => {
    switch (status) {
      case 'verified': return '[OK]';
      case 'failed': case 'rejected': return '[!]';
      case 'processing': case 'uploaded': return '[...]';
      default: return '[?]';
    }
  };

  const getDocumentIconPath = (type) => {
    return 'DOC'; // Simplified
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-3">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-4 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-gray-700"
      >
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center space-x-3">
            <span className="text-xl font-bold">{getDocumentIconPath(document.documentType)}</span>
            <div>
              <h2 className="text-lg font-semibold text-white truncate">
                {document.originalName || document.fileName}
              </h2>
              <p className="text-sm text-gray-400">
                {document.documentType?.replace('-', ' ').toUpperCase()}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white p-1">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="mb-6">
          <div className="flex items-center space-x-2 mb-3">
            <span className={`text-sm font-bold ${getStatusColor(document.status)}`}>
              {getStatusIconText(document.status)}
            </span>
            <span className={`font-semibold ${getStatusColor(document.status)} capitalize`}>
              {document.status === 'verified' ? 'Accepted' :
                (document.status === 'processing' || document.status === 'uploaded') ? 'Processing' : 'Rejected'}
            </span>
          </div>

          {document.verificationResult?.confidence !== null && (
            <div>
              <p className="text-sm text-gray-400 mb-2">
                Confidence: {Math.round(document.verificationResult.confidence || 0)}%
              </p>
              <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all duration-500"
                  style={{ width: `${document.verificationResult.confidence || 0}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white mb-3">Document Details</h3>
            <div className="space-y-3">
              <div>
                <label className="text-sm text-gray-400">Original Name</label>
                <p className="text-white">{document.originalName || document.fileName || 'Unknown'}</p>
              </div>
              <div>
                <label className="text-sm text-gray-400">File Size</label>
                <p className="text-white">{(document.fileSize / 1024 / 1024).toFixed(2)} MB</p>
              </div>
              <div>
                <label className="text-sm text-gray-400">Upload Date</label>
                <p className="text-white">{new Date(document.createdAt || document.uploadedAt).toLocaleString()}</p>
              </div>
              <div>
                <label className="text-sm text-gray-400">Document Type</label>
                <p className="text-white capitalize">{document.documentType?.replace('-', ' ') || 'Unknown'}</p>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white mb-3">Verification Details</h3>
            <div className="space-y-3">
              <div>
                <label className="text-sm text-gray-400">Status</label>
                <p className={`font-medium capitalize ${getStatusColor(document.status)}`}>
                  {document.status}
                </p>
              </div>
              {document.verificationResult?.confidence && (
                <div>
                  <label className="text-sm text-gray-400">Score</label>
                  <p className="text-white">{Math.round(document.verificationResult.confidence || 0)}%</p>
                </div>
              )}
              <div>
                <label className="text-sm text-gray-400">Security</label>
                <p className="text-green-400">Secure</p>
              </div>
            </div>
          </div>
        </div>

        {document.extractedData && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">Extracted Data</h3>
            <div className="bg-gray-800 rounded-lg p-4">
              <pre className="text-gray-300 text-sm whitespace-pre-wrap">
                {JSON.stringify(document.extractedData, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {document.verificationResult?.findings && document.verificationResult.findings.length > 0 && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">Forensic Report</h3>
            <div className="bg-gray-800 rounded-xl p-4 border border-blue-500/30 overflow-hidden relative">
              <div className="flex items-center justify-between mb-4 bg-gray-900/50 p-3 rounded-lg">
                <div>
                  <span className="text-xs text-blue-400 uppercase font-bold tracking-widest">Score</span>
                  <div className="text-3xl font-black text-white">{document.verificationResult.confidence || 0}<span className="text-sm text-gray-500">/100</span></div>
                </div>
                <div className="text-right">
                  <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase ${document.verificationResult.verdict === 'ACCEPT' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                    {document.verificationResult.verdict === 'ACCEPT' ? 'AUTHENTIC' : 'SUSPICIOUS'}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-gray-900/30 p-2 rounded border border-gray-700/50">
                  <span className="text-[10px] text-gray-400 uppercase">ELA Variance</span>
                  <div className={`text-sm font-bold ${document.verificationResult.forensics?.ela_score > 50 ? 'text-red-400' : 'text-green-400'}`}>
                    {document.verificationResult.forensics?.ela_score || 0}
                  </div>
                </div>
                <div className="bg-gray-900/30 p-2 rounded border border-gray-700/50">
                  <span className="text-[10px] text-gray-400 uppercase">Screen Det.</span>
                  <div className={`text-sm font-bold ${document.verificationResult.forensics?.is_rephoto ? 'text-red-400' : 'text-green-400'}`}>
                    {document.verificationResult.forensics?.is_rephoto ? 'YES' : 'NO'}
                  </div>
                </div>
              </div>

              <h4 className="text-xs font-bold text-gray-400 uppercase mb-2">Signals</h4>
              <ul className="space-y-2">
                {document.verificationResult.findings.map((finding, idx) => (
                  <li key={idx} className="flex items-start text-sm bg-gray-900/40 p-2 rounded border-l-2 border-blue-500/50">
                    <span className="text-gray-200">{finding}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        <div className="mb-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-white">Extracted Text (OCR)</h3>
            <button
              onClick={() => setShowOCRText(!showOCRText)}
              className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-sm py-1 px-3 rounded-lg border border-blue-500/30"
            >
              {showOCRText ? 'Hide' : 'Show'}
            </button>
          </div>

          {showOCRText && (
            <div className="bg-gray-800 rounded-lg p-4">
              {ocrLoading && (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
                  <span className="ml-3 text-gray-400">Processing...</span>
                </div>
              )}

              {ocrError && (
                <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4">
                  <span className="text-red-300 text-sm">Error: {ocrError}</span>
                </div>
              )}

              {ocrData && !ocrLoading && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm border-b border-gray-700 pb-2">
                    <span className="text-gray-400">Confidence: {Math.round(ocrData.confidence * 100)}%</span>
                    <span className="text-gray-400">
                      {ocrData.cached ? 'Cached' : 'Fresh scan'}
                    </span>
                  </div>

                  {ocrData.extractedText ? (
                    <div>
                      <h4 className="text-sm font-semibold text-gray-300 mb-2">Extracted Text:</h4>
                      <div className="bg-gray-900 rounded p-3 max-h-64 overflow-y-auto">
                        <pre className="text-gray-300 text-sm whitespace-pre-wrap">
                          {ocrData.extractedText}
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-gray-900/50 border border-gray-600 rounded-lg p-4 text-center">
                      <span className="text-gray-400">No text could be extracted from this document</span>
                    </div>
                  )}

                  {ocrData.extractedData && Object.keys(ocrData.extractedData).length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold text-gray-300 mb-2">Structured Data:</h4>
                      <div className="bg-gray-900 rounded p-3">
                        <pre className="text-gray-300 text-xs whitespace-pre-wrap">
                          {JSON.stringify(ocrData.extractedData, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}

                  {ocrData.error && (
                    <button
                      onClick={fetchOCRData}
                      className="w-full bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-sm py-2 px-4 rounded-lg border border-blue-500/30"
                    >
                      Retry
                    </button>
                  )}
                </div>
              )}

              {!ocrData && !ocrLoading && !ocrError && (
                <div className="text-center py-8">
                  <button
                    onClick={fetchOCRData}
                    className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-sm py-2 px-4 rounded-lg border border-blue-500/30"
                  >
                    Extract Text
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {document.verificationResult?.analysisDetails?.aiAnalysis?.details && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">AI Analysis</h3>
            <div className="bg-gray-800 rounded-lg p-4 border border-blue-500/30">

              <div className="flex items-center justify-between mb-4 border-b border-gray-700 pb-3">
                <div>
                  <h4 className="text-sm text-gray-400 uppercase tracking-wider font-semibold">Verdict</h4>
                  <div className={`text-xl font-bold capitalize mt-1 ${document.verificationResult.analysisDetails.aiAnalysis.details.verdict === 'authentic' ? 'text-green-400' :
                    document.verificationResult.analysisDetails.aiAnalysis.details.verdict === 'fake' ? 'text-red-400' :
                      'text-yellow-400'
                    }`}>
                    {document.verificationResult.analysisDetails.aiAnalysis.details.verdict}
                  </div>
                </div>
                <div className="text-right">
                  <h4 className="text-sm text-gray-400 uppercase tracking-wider font-semibold">Confidence</h4>
                  <div className="text-xl font-bold text-white mt-1">
                    {(document.verificationResult.analysisDetails.aiAnalysis.details.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              <div className="mb-4">
                <h4 className="text-sm text-gray-400 font-semibold mb-2">Reasons</h4>
                <ul className="space-y-2">
                  {document.verificationResult.analysisDetails.aiAnalysis.details.reasons?.map((reason, idx) => (
                    <li key={idx} className="text-sm text-gray-300 bg-gray-900/50 p-2 rounded">
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>

              {document.verificationResult.analysisDetails.aiAnalysis.details.checks && (
                <div>
                  <h4 className="text-sm text-gray-400 font-semibold mb-2">Checks</h4>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(document.verificationResult.analysisDetails.aiAnalysis.details.checks).map(([check, passed]) => (
                      <div key={check} className={`p-2 rounded text-xs font-medium border ${passed ? 'bg-green-900/20 text-green-300 border-green-900/30' :
                        'bg-red-900/20 text-red-300 border-red-900/30'
                        }`}>
                        <span className="capitalize">{check.replace(/_/g, ' ')}: {passed ? 'PASS' : 'FAIL'}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {document.verificationResult && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">Analysis Report</h3>

            <div className="bg-gray-800 rounded-lg p-4 mb-4">
              <h4 className="text-md font-semibold text-blue-400 mb-2">Assessments</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-300">Score:</span>
                  <span className="text-white font-semibold">{document.verificationResult.confidence || 0}%</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-300">Status:</span>
                  <span className={`font-semibold capitalize ${getStatusColor(document.status)}`}>
                    {document.status}
                  </span>
                </div>
              </div>
            </div>

            {document.verificationResult.analysisDetails?.anomalies && document.verificationResult.analysisDetails.anomalies.length > 0 && (
              <div className="bg-red-900 bg-opacity-50 rounded-lg p-4 mb-4 border border-red-500">
                <h4 className="text-md font-semibold text-red-400 mb-2">Alerts</h4>
                <div className="space-y-2">
                  {document.verificationResult.analysisDetails.anomalies.map((anomaly, index) => (
                    <div key={index} className="text-sm text-red-300">• {anomaly}</div>
                  ))}
                </div>
              </div>
            )}

            <div className={`rounded-lg p-4 border ${document.status === 'verified' ? 'bg-green-900 bg-opacity-50 border-green-500' :
              'bg-red-900 bg-opacity-50 border-red-500'
              }`}>
              <h4 className="text-md font-semibold text-white mb-2">Final Recommendation</h4>
              <div className="text-sm">
                {document.status === 'verified' ? (
                  <p className="text-green-300">ACCEPT - Document appears authentic</p>
                ) : (
                  <p className="text-red-300">REJECT - Verification failed</p>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="flex space-x-3">
          <button className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg transition">
            Download
          </button>
          <button className="flex-1 bg-green-600 hover:bg-green-700 text-white py-2 rounded-lg transition">
            Share
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default DocumentDetailsModal;
