import React from 'react';

const DocumentCard = ({ document, onUpdate }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'verified': return 'text-green-400';
      case 'processing':
      case 'needs_review': return 'text-yellow-400';
      case 'rejected':
      case 'failed': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIconText = (status) => {
    switch (status) {
      case 'verified': return '[OK]';
      case 'processing': return '[...]';
      case 'needs_review': return '[!]';
      case 'rejected':
      case 'failed': return '[X]';
      default: return '[?]';
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-lg p-4 border border-white/10 hover:bg-white/10 transition-all duration-200">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-bold">{getStatusIconText(document.status)}</span>
          <div>
            <h3 className="font-semibold text-white text-sm truncate max-w-32">
              {document.originalName || document.filename || 'Document'}
            </h3>
            <p className="text-xs text-gray-400">
              {document.documentType || 'Unknown'}
            </p>
          </div>
        </div>
        <span className={`text-xs font-bold uppercase ${getStatusColor(document.status)}`}>
          {document.status}
        </span>
      </div>

      <div className="space-y-2 mb-4">
        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-400">Confidence:</span>
          <span className="text-white font-medium">{document.confidence || 0}%</span>
        </div>

        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-400">Uploaded:</span>
          <span className="text-white">{formatDate(document.createdAt || document.uploadDate)}</span>
        </div>

        {document.fileSize && (
          <div className="flex justify-between items-center text-xs">
            <span className="text-gray-400">Size:</span>
            <span className="text-white">{(document.fileSize / 1024 / 1024).toFixed(2)} MB</span>
          </div>
        )}
      </div>

      {document.status === 'processing' && (
        <div className="mb-4">
          <div className="w-full bg-gray-700 rounded-full h-1.5">
            <div
              className="bg-blue-500 h-1.5 rounded-full transition-all duration-500"
              style={{ width: `${document.progress || 50}%` }}
            ></div>
          </div>
        </div>
      )}

      <div className="flex space-x-2">
        <button
          onClick={() => console.log('View:', document.id)}
          className="flex-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 text-xs py-2 px-3 rounded border border-blue-600/30"
        >
          View Details
        </button>

        {document.status === 'verified' && (
          <button
            onClick={() => console.log('Download:', document.id)}
            className="bg-green-600/20 hover:bg-green-600/30 text-green-400 text-xs py-2 px-3 rounded border border-green-600/30"
          >
            DL
          </button>
        )}
      </div>
    </div>
  );
};

export default DocumentCard;
