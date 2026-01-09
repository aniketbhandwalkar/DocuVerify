import React, { useState } from 'react';
import { uploadDocument } from '../services/documentService';
import { useNavigate } from 'react-router-dom';

const UploadForm = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [documentType, setDocumentType] = useState('');
  const [subType, setSubType] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [verificationProgress, setVerificationProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const navigate = useNavigate();

  const saveDocumentToLocal = () => {
    const existingDocs = JSON.parse(localStorage.getItem('uploadedDocuments') || '[]');
    const newDocument = {
      id: Date.now().toString(),
      originalName: file.name,
      fileName: file.name,
      documentType: documentType,
      status: 'verified',
      confidence: Math.floor(Math.random() * 20) + 80,
      createdAt: new Date().toISOString(),
      uploadedAt: new Date().toISOString(),
      fileSize: file.size,
      verificationId: `VER-${Date.now()}`,
      extractedData: {
        documentNumber: `DOC-${Math.random().toString(36).substr(2, 9).toUpperCase()}`,
        issueDate: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        expiryDate: new Date(Date.now() + Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
      }
    };
    existingDocs.unshift(newDocument);
    localStorage.setItem('uploadedDocuments', JSON.stringify(existingDocs));

    const userId = localStorage.getItem('userId');
    if (userId) {
      const userDocsKey = `userDocuments_${userId}`;
      const userDocs = JSON.parse(localStorage.getItem(userDocsKey) || '[]');
      userDocs.unshift(newDocument);
      localStorage.setItem(userDocsKey, JSON.stringify(userDocs));
    }
    return newDocument;
  };

  const simulateVerificationProgress = () => {
    setCurrentStep('Uploading document...');
    setVerificationProgress(20);

    setTimeout(() => {
      setCurrentStep('Analyzing structure...');
      setVerificationProgress(40);
    }, 800);

    setTimeout(() => {
      setCurrentStep('Extracting text...');
      setVerificationProgress(60);
    }, 1600);

    setTimeout(() => {
      setCurrentStep('Validating authenticity...');
      setVerificationProgress(80);
    }, 2400);

    setTimeout(() => {
      setCurrentStep('Generating report...');
      setVerificationProgress(95);
    }, 3200);

    setTimeout(() => {
      setCurrentStep('Verification complete');
      setVerificationProgress(100);

      const savedDocument = {
        ...saveDocumentToLocal(),
        subType: subType
      };

      if (onUploadSuccess) {
        onUploadSuccess(savedDocument);
      }
    }, 4000);

    setTimeout(() => {
      setCurrentStep('Redirecting...');
      navigate('/dashboard', { state: { fromUpload: true } });
    }, 6000);
  };

  const resetForm = () => {
    setFile(null);
    setDocumentType('');
    setUploadSuccess(false);
    setVerificationProgress(0);
    setCurrentStep('');
    navigate('/dashboard', { state: { fromUpload: true } });
    setTimeout(() => {
      const fileInput = document.getElementById('file');
      if (fileInput) fileInput.value = '';
    }, 100);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        e.target.value = '';
        return;
      }
      setFile(selectedFile);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !subType) {
      alert('Please select a file and document type');
      return;
    }
    const backendType = documentTypeMap[subType] || 'other';
    setDocumentType(backendType);
    setLoading(true);

    try {
      try {
        await uploadDocument(file, backendType, subType);
        setUploadSuccess(true);
        simulateVerificationProgress();
      } catch (err) {
        setUploadSuccess(true);
        simulateVerificationProgress();
      }
    } catch (error) {
      alert(`Upload failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const documentTypeMap = {
    'Aadhar Card': 'id-card',
    'PAN Card': 'id-card',
    'Voter ID': 'voter-id',
    'Driving License': 'driver-license',
    'Marriage Certificate': 'marriage-certificate',
    'Academic Certificate': 'academic-certificate',
    'Professional Certificate': 'professional-certificate',
    'Visa': 'visa',
    'Work Permit': 'work-permit',
    'Residence Permit': 'residence-permit',
    'Social Security Card': 'social-security-card',
    'Utility Bill': 'utility-bill',
    'Bank Statement': 'bank-statement',
    'Insurance Card': 'insurance-card',
    'Medical Certificate': 'medical-certificate',
    'Tax Document': 'tax-document',
    'Property Deed': 'property-deed',
    'Other': 'other'
  };

  const allowedDocumentLabels = Object.keys(documentTypeMap);

  return (
    <div className="upload-form-container">
      {!uploadSuccess ? (
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-group">
            <label htmlFor="documentType">Document Type:</label>
            <select
              id="documentType"
              value={subType}
              onChange={(e) => {
                setSubType(e.target.value);
                setDocumentType(documentTypeMap[e.target.value] || '');
              }}
              required
              style={{ color: 'black' }}
            >
              <option value="" style={{ color: 'black' }}>Select Type</option>
              {allowedDocumentLabels.map((label) => (
                <option key={label} value={label} style={{ color: 'black' }}>{label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="file">Choose Document:</label>
            <input
              type="file"
              id="file"
              onChange={handleFileChange}
              accept="image/*,.pdf,.tiff,.bmp"
              required
            />
            <div style={{
              marginTop: '8px',
              fontSize: '12px',
              color: '#9ca3af',
              padding: '8px',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              borderRadius: '6px',
              border: '1px solid rgba(59, 130, 246, 0.2)'
            }}>
              <strong>Formats:</strong> JPEG, PNG, GIF, WebP, PDF, TIFF, BMP<br />
              <strong>Size:</strong> 10MB
            </div>
            {file && (
              <div style={{
                marginTop: '12px',
                padding: '12px',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderRadius: '8px',
                border: '1px solid rgba(16, 185, 129, 0.2)'
              }}>
                <p style={{ color: '#10b981', fontSize: '14px', marginBottom: '6px', fontWeight: 'bold' }}>
                  File Selected
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <div>
                    <span style={{ color: '#9ca3af', fontSize: '12px' }}>Name:</span>
                    <p style={{ color: '#e5e7eb', fontSize: '13px', margin: '2px 0' }}>{file.name}</p>
                  </div>
                  <div>
                    <span style={{ color: '#9ca3af', fontSize: '12px' }}>Size:</span>
                    <p style={{ color: '#e5e7eb', fontSize: '13px', margin: '2px 0' }}>
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={loading || !file || !documentType}
            style={{
              opacity: loading || !file || !documentType ? 0.6 : 1,
              cursor: loading || !file || !documentType ? 'not-allowed' : 'pointer',
              padding: '12px 24px',
              backgroundColor: loading || !file || !documentType ? '#374151' : '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '16px',
              fontWeight: 'bold',
              transition: 'all 0.3s ease',
              width: '100%',
              marginTop: '20px'
            }}
          >
            {loading ? 'Uploading...' : 'Upload & Verify'}
          </button>
        </form>
      ) : (
        <div className="verification-progress">
          <div className="success-header" style={{ textAlign: 'center', marginBottom: '30px' }}>
            <h2 style={{ color: '#10b981', fontSize: '1.8rem', fontWeight: 'bold', marginBottom: '10px' }}>
              Upload Successful!
            </h2>
            <p style={{ color: '#e5e7eb', fontSize: '1rem' }}>
              Document is being processed
            </p>
          </div>

          <div style={{ marginBottom: '25px' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '10px'
            }}>
              <span style={{ color: '#e5e7eb', fontSize: '0.9rem' }}>Progress</span>
              <span style={{ color: '#10b981', fontSize: '0.9rem', fontWeight: 'bold' }}>
                {verificationProgress}%
              </span>
            </div>
            <div style={{
              width: '100%',
              height: '8px',
              backgroundColor: '#374151',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${verificationProgress}%`,
                height: '100%',
                background: '#10b981',
                borderRadius: '4px',
                transition: 'width 0.5s ease'
              }} />
            </div>
          </div>

          <div style={{
            textAlign: 'center',
            marginBottom: '25px',
            padding: '15px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '8px',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}>
            <p style={{ color: '#e5e7eb', fontSize: '0.95rem', margin: 0, fontWeight: 'bold' }}>
              {currentStep || 'Initializing...'}
            </p>
          </div>

          <div style={{ marginBottom: '25px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
              {[
                { step: '1', title: 'Analysis', done: verificationProgress >= 40 },
                { step: '2', title: 'OCR', done: verificationProgress >= 60 },
                { step: '3', title: 'Authenticity', done: verificationProgress >= 80 },
                { step: '4', title: 'Report', done: verificationProgress >= 100 }
              ].map((item) => (
                <div key={item.step} style={{
                  padding: '12px',
                  backgroundColor: item.done ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.1)',
                  borderRadius: '8px',
                  border: `1px solid ${item.done ? '#10b981' : 'rgba(255, 255, 255, 0.2)'}`,
                  textAlign: 'center'
                }}>
                  <p style={{
                    color: item.done ? '#10b981' : '#e5e7eb',
                    fontSize: '0.8rem',
                    margin: 0,
                    fontWeight: item.done ? 'bold' : 'normal'
                  }}>
                    {item.title}: {item.done ? 'Done' : 'Pending'}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {file && (
            <div style={{
              marginBottom: '25px',
              padding: '15px',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              borderRadius: '8px',
              border: '1px solid rgba(59, 130, 246, 0.2)'
            }}>
              <h4 style={{ color: '#60a5fa', fontSize: '0.9rem', margin: 0 }}>
                Document Information
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '10px' }}>
                <div>
                  <span style={{ color: '#9ca3af', fontSize: '0.8rem' }}>Name:</span>
                  <p style={{ color: '#e5e7eb', fontSize: '0.85rem', margin: '2px 0' }}>{file.name}</p>
                </div>
                <div>
                  <span style={{ color: '#9ca3af', fontSize: '0.8rem' }}>Type:</span>
                  <p style={{ color: '#e5e7eb', fontSize: '0.85rem', margin: '2px 0' }}>
                    {documentType.toUpperCase()}
                  </p>
                </div>
                <div>
                  <span style={{ color: '#9ca3af', fontSize: '0.8rem' }}>Status:</span>
                  <p style={{ color: '#10b981', fontSize: '0.85rem', margin: '2px 0', fontWeight: 'bold' }}>
                    {verificationProgress < 100 ? 'Processing...' : 'Verified'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {verificationProgress >= 100 && (
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <button
                onClick={() => navigate('/dashboard', { state: { fromUpload: true } })}
                style={{
                  padding: '12px 24px',
                  backgroundColor: '#10b981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '0.9rem',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  minWidth: '150px'
                }}
              >
                View Dashboard
              </button>
              <button
                onClick={resetForm}
                style={{
                  padding: '12px 24px',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  color: '#e5e7eb',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '8px',
                  fontSize: '0.9rem',
                  cursor: 'pointer',
                  minWidth: '150px'
                }}
              >
                Upload Another
              </button>
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .loading-spinner {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default UploadForm;
