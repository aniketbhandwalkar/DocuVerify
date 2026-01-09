import React, { useState, useEffect, useContext } from 'react';
import { getDocuments } from '../services/documentService';
import { useNavigate, useLocation } from 'react-router-dom';
import { AuthContext } from '../contexts/AuthContext';
import { useCategory } from '../contexts/CategoryContext';
import DocumentDetailsModal from '../components/DocumentDetailsModal';

const Dashboard = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    total: 0,
    verified: 0,
    rejected: 0
  });
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [showAllDocuments, setShowAllDocuments] = useState(false);
  const [documentFilter, setDocumentFilter] = useState('all'); // all, verified, rejected
  const [documentSort, setDocumentSort] = useState('newest'); // newest, oldest, name
  const [autoRefreshing, setAutoRefreshing] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useContext(AuthContext);
  const { selectCategory, clearCategory, categories } = useCategory();

  useEffect(() => {
    fetchDocuments();
  }, []);

  // Listen for navigation from upload page to refresh documents
  useEffect(() => {
    if (location.state?.fromUpload) {
      fetchDocuments();
      // Clear the state to prevent multiple refreshes
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);



  // Function to get user name from multiple sources
  const getUserName = () => {
    // First try to get from AuthContext
    if (user?.name && user.name !== 'undefined') {
      return user.name;
    }

    // Then try to get from localStorage user object
    const storedUser = localStorage.getItem('user');
    if (storedUser && storedUser !== 'undefined') {
      try {
        const parsedUser = JSON.parse(storedUser);
        // Handle both direct user object and nested user object
        const userData = parsedUser.user || parsedUser;

        if (userData.name && userData.name !== 'undefined') {
          return userData.name;
        }
        // Try email from parsed user
        if (userData.email && userData.email !== 'undefined') {
          return userData.email.split('@')[0];
        }
      } catch (e) {
        console.log('Error parsing stored user:', e);
      }
    }

    // Try to get from AuthContext email
    if (user?.email && user.email !== 'undefined') {
      return user.email.split('@')[0];
    }

    // If we still have a token, the user is logged in - extract from token email
    const token = localStorage.getItem('token');
    if (token) {
      // Try to get email from any available source since user is authenticated
      const email = user?.email || 'yuvibhatkar702@gmail.com';
      return email.split('@')[0];
    }

    return 'User';
  };

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      let documentsData = [];

      // Try to get real documents from API first
      try {
        const token = localStorage.getItem('token');
        if (token) {
          const response = await getDocuments();
          console.log('[Dashboard] getDocuments() API response:', response);
          if (response && response.success && response.data) {
            documentsData = response.data;
          } else if (response && Array.isArray(response)) {
            documentsData = response;
          }
        }
      } catch (fetchError) {
        console.log('[Dashboard] API fetch error:', fetchError.message);
        // If API fails, check for locally stored documents for this user
        const userId = user?.id || localStorage.getItem('userId');
        if (userId) {
          const userDocs = JSON.parse(localStorage.getItem(`userDocuments_${userId}`) || '[]');
          documentsData = userDocs;
        }
      }

      // Sort by creation date (newest first)
      if (documentsData.length > 0) {
        documentsData.sort((a, b) => new Date(b.createdAt || b.uploadedAt) - new Date(a.createdAt || a.uploadedAt));
      }

      console.log('[Dashboard] Final documentsData to set:', documentsData);
      setDocuments(documentsData);

      // Calculate stats based on actual documents
      const stats = documentsData.reduce((acc, doc) => {
        acc.total++;
        if (doc.status === 'verified') acc.verified++;
        else if (doc.status === 'rejected' || doc.status === 'failed') acc.rejected++;
        return acc;
      }, { total: 0, verified: 0, rejected: 0 });

      setStats(stats);
    } catch (error) {
      console.error('[Dashboard] Error in fetchDocuments:', error);
      setError('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadClick = (categoryId = null) => {
    if (categoryId) {
      selectCategory(categoryId);
    } else {
      clearCategory();
    }
    navigate('/upload');
  };

  // Function to refresh documents (can be called from child components)
  const refreshDocuments = () => {
    fetchDocuments();
  };

  // Modal handler functions
  const handleViewDetails = (document) => {
    setSelectedDocument(document);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedDocument(null);
  };

  // Logout handler
  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  // Filter and sort documents
  const getFilteredAndSortedDocuments = () => {
    let filteredDocs = [...documents];
    // Enable filter logic for document status
    if (documentFilter !== 'all') {
      filteredDocs = filteredDocs.filter(doc => {
        switch (documentFilter) {
          case 'verified':
            return doc.status === 'verified';
          case 'rejected':
            return doc.status === 'rejected' || doc.status === 'failed';
          default:
            return true;
        }
      });
    }
    // Apply sorting
    filteredDocs.sort((a, b) => {
      switch (documentSort) {
        case 'oldest':
          return new Date(a.createdAt || a.uploadedAt) - new Date(b.createdAt || b.uploadedAt);
        case 'name':
          return (a.originalName || a.fileName || '').localeCompare(b.originalName || b.fileName || '');
        case 'newest':
        default:
          return new Date(b.createdAt || b.uploadedAt) - new Date(a.createdAt || a.uploadedAt);
      }
    });
    return filteredDocs;
  };

  const filteredDocuments = getFilteredAndSortedDocuments();

  const DocumentCard = ({ document }) => {
    const getStatusColor = (status) => {
      switch (status) {
        case 'verified': return 'text-green-400';
        case 'rejected': case 'failed': return 'text-red-400';
        case 'processing': case 'uploaded': return 'text-yellow-400';
        default: return 'text-gray-400';
      }
    };

    const getStatusIcon = (status) => {
      switch (status) {
        case 'verified': return '✅';
        case 'rejected': case 'failed': return '❌';
        case 'processing': case 'uploaded': return '⏳';
        default: return '📄';
      }
    };

    const formatFileSize = (bytes) => {
      if (!bytes) return 'Unknown';
      const mb = bytes / 1024 / 1024;
      return `${mb.toFixed(1)} MB`;
    };

    return (
      <div className="bg-white/8 backdrop-blur-sm rounded-xl p-3 border border-white/10 hover:bg-white/12 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/10">
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center space-x-2 flex-1 min-w-0">
            <span className="text-base flex-shrink-0">{getStatusIcon(document.status)}</span>
            <div className="min-w-0 flex-1">
              <h3 className="font-medium text-white text-xs truncate">
                {document.subType || document.originalName || document.fileName || 'Document'}
              </h3>
              <p className="text-xs text-gray-400 truncate">
                {document.subType
                  ? document.documentType?.replace('-', ' ').toUpperCase() + ' - ' + document.subType
                  : document.documentType?.replace('-', ' ').toUpperCase() || 'Unknown'}
              </p>
            </div>
          </div>
          <span className={`text-xs font-medium px-2 py-1 rounded-full bg-opacity-20 ${getStatusColor(document.status)} ${document.status === 'verified' ? 'bg-green-500' :
              (document.status === 'processing' || document.status === 'uploaded') ? 'bg-yellow-500' : 'bg-red-500'
            }`}>
            {document.status === 'verified' ? 'Accepted' :
              (document.status === 'processing' || document.status === 'uploaded') ? 'Processing' : 'Rejected'}
          </span>
        </div>

        <div className="space-y-1 mb-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-gray-400">Confidence:</span>
            <span className="text-xs font-medium text-white">
              {document.verificationResult?.confidence || 0}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-gray-400">Date:</span>
            <span className="text-xs text-white">
              {new Date(document.createdAt || document.uploadedAt).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric'
              })}
            </span>
          </div>
          {document.fileSize && (
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-400">Size:</span>
              <span className="text-xs text-white">{formatFileSize(document.fileSize)}</span>
            </div>
          )}
        </div>

        <div className="flex space-x-2">
          <button
            onClick={() => handleViewDetails(document)}
            className="flex-1 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-xs py-2 px-3 rounded-lg border border-blue-500/30 transition-all duration-200 hover:border-blue-500/50"
          >
            View
          </button>
          <button
            onClick={handleUploadClick}
            className="bg-green-500/20 hover:bg-green-500/30 text-green-400 text-xs py-2 px-3 rounded-lg border border-green-500/30 transition-all duration-200 hover:border-green-500/50"
          >
            📤
          </button>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-white">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={fetchDocuments}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Mobile-First Header */}
      <div className="bg-black/20 backdrop-blur-sm border-b border-white/10 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-3 sm:px-6">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">📄</span>
              </div>
              <div>
                <h1 className="text-white font-semibold text-sm">
                  Hello, {getUserName()}!
                </h1>
                <p className="text-gray-400 text-xs hidden sm:block">
                  Document Verification Dashboard
                </p>
              </div>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden bg-white/10 p-2 rounded-lg text-white"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            {/* Desktop Action Buttons */}
            <div className="hidden md:flex items-center space-x-2">
              <button
                onClick={refreshDocuments}
                className="bg-white/10 hover:bg-white/20 text-white font-medium py-2 px-4 rounded-lg border border-white/20 transition-all duration-200 text-sm"
              >
                🔄
              </button>

              <button
                onClick={() => navigate('/profile')}
                className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 font-medium py-2 px-4 rounded-lg border border-purple-500/30 transition-all duration-200 text-sm"
              >
                ⚙️
              </button>

              <button
                onClick={handleLogout}
                className="bg-red-500/20 hover:bg-red-500/30 text-red-400 font-medium py-2 px-4 rounded-lg border border-red-500/30 transition-all duration-200 text-sm"
              >
                🚪
              </button>
            </div>
          </div>

          {/* Mobile Menu */}
          {isMobileMenuOpen && (
            <div className="md:hidden bg-black/40 backdrop-blur-sm border-t border-white/10 py-3">
              <div className="flex flex-col space-y-2">
                <button
                  onClick={() => {
                    refreshDocuments();
                    setIsMobileMenuOpen(false);
                  }}
                  className="bg-white/10 text-white font-medium py-2 px-4 rounded-lg text-sm flex items-center space-x-2"
                >
                  <span>🔄</span>
                  <span>Refresh Documents</span>
                </button>

                <button
                  onClick={() => {
                    navigate('/profile');
                    setIsMobileMenuOpen(false);
                  }}
                  className="bg-purple-500/20 text-purple-400 font-medium py-2 px-4 rounded-lg text-sm flex items-center space-x-2"
                >
                  <span>⚙️</span>
                  <span>Profile & Settings</span>
                </button>

                <button
                  onClick={() => {
                    handleLogout();
                    setIsMobileMenuOpen(false);
                  }}
                  className="bg-red-500/20 text-red-400 font-medium py-2 px-4 rounded-lg text-sm flex items-center space-x-2"
                >
                  <span>🚪</span>
                  <span>Logout</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-3 sm:px-6 py-4">
        {/* Compact Stats Cards */}
        <div className="grid grid-cols-3 gap-3 mb-6">
          <div className="bg-white/8 backdrop-blur-sm rounded-xl p-4 border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-xs font-medium">Total</p>
                <p className="text-xl font-bold text-white">{stats.total}</p>
              </div>
              <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <span className="text-sm">📊</span>
              </div>
            </div>
          </div>

          <div className="bg-white/8 backdrop-blur-sm rounded-xl p-4 border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-xs font-medium">Accepted</p>
                <p className="text-xl font-bold text-green-400">{stats.verified}</p>
              </div>
              <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <span className="text-sm">✅</span>
              </div>
            </div>
          </div>

          <div className="bg-white/8 backdrop-blur-sm rounded-xl p-4 border border-white/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-xs font-medium">Rejected</p>
                <p className="text-xl font-bold text-red-400">{stats.rejected}</p>
              </div>
              <div className="w-8 h-8 bg-red-500/20 rounded-lg flex items-center justify-center">
                <span className="text-sm">❌</span>
              </div>
            </div>
          </div>
        </div>

        {/* Documents Section */}
        <div className="bg-white/8 backdrop-blur-sm rounded-xl border border-white/10 overflow-hidden">
          <div className="p-4 border-b border-white/10">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
              <div className="flex items-center space-x-3">
                <h2 className="text-lg font-semibold text-white">
                  {showAllDocuments ? 'All Documents' : 'Recent Documents'}
                </h2>
                <span className="text-xs text-gray-400">
                  {filteredDocuments.length} of {documents.length} document{documents.length !== 1 ? 's' : ''}
                </span>
              </div>

              <div className="flex items-center space-x-2">
                {/* Filter Dropdown */}
                {showAllDocuments && (
                  <>
                    <select
                      value={documentFilter}
                      onChange={(e) => setDocumentFilter(e.target.value)}
                      className="bg-white/10 text-white text-xs py-1 px-2 rounded border border-white/20 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="all" className="bg-gray-800">All Status</option>
                      <option value="verified" className="bg-gray-800">✅ Accepted</option>
                      <option value="rejected" className="bg-gray-800">❌ Rejected</option>
                    </select>

                    <select
                      value={documentSort}
                      onChange={(e) => setDocumentSort(e.target.value)}
                      className="bg-white/10 text-white text-xs py-1 px-2 rounded border border-white/20 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="newest" className="bg-gray-800">📅 Newest</option>
                      <option value="oldest" className="bg-gray-800">📅 Oldest</option>
                      <option value="name" className="bg-gray-800">🔤 Name</option>
                    </select>
                  </>
                )}

                {/* Manual Refresh Button */}
                <button
                  onClick={() => {
                    fetchDocuments();
                  }}
                  disabled={loading}
                  className="bg-green-500/20 hover:bg-green-500/30 disabled:bg-gray-500/20 text-green-400 disabled:text-gray-500 text-xs py-1 px-3 rounded-lg border border-green-500/30 disabled:border-gray-500/30 transition-all duration-200 hover:border-green-500/50 flex items-center space-x-1"
                >
                  <svg className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span>{loading ? 'Refreshing...' : 'Refresh'}</span>
                </button>

                {documents.length > 8 && (
                  <button
                    onClick={() => {
                      setShowAllDocuments(!showAllDocuments);
                      if (!showAllDocuments) {
                        setDocumentFilter('all');
                        setDocumentSort('newest');
                      }
                    }}
                    className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-xs py-1 px-3 rounded-lg border border-blue-500/30 transition-all duration-200 hover:border-blue-500/50"
                  >
                    {showAllDocuments ? '📋 Show Recent' : '📄 View All'}
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="p-4">
            {documents.length === 0 ? (
              <div className="text-center py-8">
                <div className="w-16 h-16 bg-gray-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">📄</span>
                </div>
                <h3 className="text-lg font-medium text-gray-300 mb-2">No documents yet</h3>
                <p className="text-gray-400 text-sm mb-4">Upload your first document to get started</p>
                <button
                  onClick={handleUploadClick}
                  className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white font-medium py-2 px-4 rounded-lg transition-all duration-200 text-sm"
                >
                  🚀 Upload Document
                </button>
              </div>
            ) : filteredDocuments.length === 0 ? (
              <div className="text-center py-8">
                <div className="w-16 h-16 bg-gray-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">🔍</span>
                </div>
                <h3 className="text-lg font-medium text-gray-300 mb-2">No documents match filter</h3>
                <p className="text-gray-400 text-sm mb-4">Try changing your filter criteria</p>
                <button
                  onClick={() => setDocumentFilter('all')}
                  className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-xs py-2 px-4 rounded-lg border border-blue-500/30 transition-all duration-200"
                >
                  Reset Filter
                </button>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {(showAllDocuments ? filteredDocuments : filteredDocuments.slice(0, 8)).map((document) => (
                    <DocumentCard
                      key={document.id || document.fileName}
                      document={document}
                    />
                  ))}
                </div>

                {/* Show pagination info for all documents view */}
                {showAllDocuments && filteredDocuments.length > 8 && (
                  <div className="mt-6 text-center">
                    <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                      <p className="text-xs text-gray-400">
                        Showing all {filteredDocuments.length} documents
                        {documentFilter !== 'all' && ` (${documentFilter})`}
                        {documentSort !== 'newest' && ` sorted by ${documentSort}`}
                      </p>
                      <button
                        onClick={() => {
                          setShowAllDocuments(false);
                          setDocumentFilter('all');
                          setDocumentSort('newest');
                        }}
                        className="mt-2 text-blue-400 hover:text-blue-300 text-xs underline"
                      >
                        ← Back to Recent
                      </button>
                    </div>
                  </div>
                )}

                {/* Show "View All" hint when showing recent */}
                {!showAllDocuments && documents.length > 8 && (
                  <div className="mt-6 text-center">
                    <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                      <p className="text-xs text-gray-400 mb-2">
                        Showing {Math.min(8, filteredDocuments.length)} of {documents.length} documents
                      </p>
                      <button
                        onClick={() => setShowAllDocuments(true)}
                        className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 hover:from-blue-500/30 hover:to-purple-500/30 text-blue-400 text-xs py-2 px-4 rounded-lg border border-blue-500/30 transition-all duration-200 flex items-center space-x-2 mx-auto"
                      >
                        <span>📄</span>
                        <span>View All {documents.length} Documents</span>
                        <span>→</span>
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Quick Actions - Mobile Optimized */}
        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(categories).map(([categoryId, category]) => {
            const colorClasses = {
              blue: {
                bg: 'bg-blue-500/20',
                hover: 'hover:bg-blue-500/30',
                text: 'text-blue-400',
                border: 'border-blue-500/30'
              },
              green: {
                bg: 'bg-green-500/20',
                hover: 'hover:bg-green-500/30',
                text: 'text-green-400',
                border: 'border-green-500/30'
              },
              purple: {
                bg: 'bg-purple-500/20',
                hover: 'hover:bg-purple-500/30',
                text: 'text-purple-400',
                border: 'border-purple-500/30'
              },
              yellow: {
                bg: 'bg-yellow-500/20',
                hover: 'hover:bg-yellow-500/30',
                text: 'text-yellow-400',
                border: 'border-yellow-500/30'
              },
              indigo: {
                bg: 'bg-indigo-500/20',
                hover: 'hover:bg-indigo-500/30',
                text: 'text-indigo-400',
                border: 'border-indigo-500/30'
              },
              teal: {
                bg: 'bg-teal-500/20',
                hover: 'hover:bg-teal-500/30',
                text: 'text-teal-400',
                border: 'border-teal-500/30'
              },
              orange: {
                bg: 'bg-orange-500/20',
                hover: 'hover:bg-orange-500/30',
                text: 'text-orange-400',
                border: 'border-orange-500/30'
              },
              red: {
                bg: 'bg-red-500/20',
                hover: 'hover:bg-red-500/30',
                text: 'text-red-400',
                border: 'border-red-500/30'
              }
            };

            const colors = colorClasses[category.color] || colorClasses.blue; // fallback to blue

            // Get appropriate button text
            const getButtonText = (categoryName) => {
              switch (categoryName) {
                case 'ID Documents': return 'Upload ID';
                case 'Certificates': return 'Upload Cert';
                case 'School Certificates': return 'Upload School';
                case 'College Transcripts': return 'Upload College';
                case 'Internship Letters': return 'Upload Internship';
                case 'Government Issued Certificates': return 'Upload Gov';
                case 'Employment Documents': return 'Upload Employment';
                case 'Other Documents': return 'Upload Doc';
                default: return 'Upload Doc';
              }
            };

            return (
              <div
                key={categoryId}
                className="bg-white/8 backdrop-blur-sm rounded-xl p-4 border border-white/10 text-center"
              >
                <div className={`w-12 h-12 ${colors.bg} rounded-full flex items-center justify-center mx-auto mb-3`}>
                  <span className="text-xl">{category.icon}</span>
                </div>
                <h3 className="text-sm font-semibold text-white mb-2">{category.name}</h3>
                <p className="text-gray-300 text-xs mb-3">{category.description}</p>
                <button
                  onClick={() => handleUploadClick(categoryId)}
                  className={`${colors.bg} ${colors.hover} ${colors.text} px-3 py-2 rounded-lg text-xs border ${colors.border} transition-all duration-200`}
                >
                  {getButtonText(category.name)}
                </button>
              </div>
            );
          })}
        </div>

        {/* Document Details Modal */}
        {isModalOpen && (
          <DocumentDetailsModal
            document={selectedDocument}
            isOpen={isModalOpen}
            onClose={handleCloseModal}
          />
        )}
      </div>
    </div>
  );
};

export default Dashboard;