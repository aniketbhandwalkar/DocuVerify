const Document = require('../models/Document');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const AIMlService = require('../services/aiMlService');

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads/documents';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const fileFilter = (req, file, cb) => {
  const allowedMimeTypes = [
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp',
    'application/pdf', 'image/tiff', 'image/bmp'
  ];
  const allowedExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf', '.tiff', '.bmp'];
  const ext = path.extname(file.originalname).toLowerCase();

  if (allowedMimeTypes.includes(file.mimetype) || allowedExtensions.includes(ext)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type.'), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 10 * 1024 * 1024 }
});

const { expandedDocumentTypes } = require('../data/documentTypes');

const uploadDocument = async (req, res) => {
  try {
    if (!req.user || !req.user.id) return res.status(401).json({ success: false, message: 'Unauthorized' });
    if (!req.file) return res.status(400).json({ success: false, message: 'No file uploaded' });

    const { documentType } = req.body;
    if (!expandedDocumentTypes.includes(documentType)) {
      return res.status(400).json({ success: false, message: `Invalid document type "${documentType}".` });
    }

    const document = new Document({
      userId: req.user.id,
      originalName: req.file.originalname,
      fileName: req.file.filename,
      filePath: req.file.path,
      documentType,
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      status: 'uploaded'
    });

    const savedDocument = await document.save();

    if (documentType === 'aadhar-card') {
      await processDocumentWithAI(savedDocument._id, savedDocument.filePath, documentType);
      const updatedDoc = await Document.findById(savedDocument._id);
      return res.status(201).json({ success: true, message: 'Document processed', data: updatedDoc });
    }

    processDocumentWithAI(savedDocument._id, savedDocument.filePath, documentType);
    res.status(201).json({ success: true, message: 'Document uploaded', data: savedDocument });

  } catch (error) {
    console.error(error);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ success: false, message: 'Upload failed', error: error.message });
  }
};

const processDocumentWithAI = async (documentId, filePath, documentType) => {
  try {
    if (documentType !== 'aadhar-card') {
      await Document.findByIdAndUpdate(documentId, {
        status: 'pending',
        processedAt: new Date()
      });
      return;
    }

    const result = await AIMlService.verifyAadhaarOffline(filePath);

    let status = 'rejected';
    let authenticity = 'fake';

    if (result.verdict === 'ACCEPT') {
      status = 'verified';
      authenticity = 'authentic';
    } else if (result.verdict === 'NEEDS_REUPLOAD') {
      status = 'pending_review';
      authenticity = 'uncertain';
    } else if (result.verdict === 'ERROR' || !result.success) {
      status = 'failed';
      authenticity = 'unknown';
    }

    console.log('\n--- ANALYSIS REPORT ---');
    console.log(`Document ID: ${documentId}`);
    console.log(`Verdict: ${result.verdict}`);
    console.log(`Confidence: ${result.confidence}%`);
    console.log('------------------------\n');

    await Document.findByIdAndUpdate(documentId, {
      status,
      processedAt: new Date(),
      verifiedAt: status === 'verified' ? new Date() : null,
      verificationResult: {
        confidence: result.confidence || 0,
        authenticity,
        verdict: result.verdict,
        reason: result.reason,
        findings: result.findings || [],
        forensics: result.forensics || {},
        analysisDetails: {
          ocrResult: {
            text: result.extracted_data?.aadhaar_number || '',
            confidence: (result.confidence || 0) / 100,
            success: result.success
          }
        }
      },
      extractedText: result.reason || '',
      extractedData: result.extracted_data || {}
    });

  } catch (err) {
    await Document.findByIdAndUpdate(documentId, {
      status: 'failed',
      error: err.message,
      processedAt: new Date()
    });
  }
};

const getDocuments = async (req, res) => {
  try {
    const documents = await Document.find({ userId: req.user.id }).sort({ createdAt: -1 }).select('-filePath');
    res.json({ success: true, data: documents });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Fetch failed', error: err.message });
  }
};

const getDocument = async (req, res) => {
  try {
    const document = await Document.findOne({ _id: req.params.id, userId: req.user.id });
    if (!document) return res.status(404).json({ success: false, message: 'Document not found' });
    res.json({ success: true, data: document });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Get failed', error: err.message });
  }
};

const verifyDocument = async (req, res) => {
  try {
    const { isValid, notes } = req.body;
    const document = await Document.findById(req.params.id);
    if (!document) return res.status(404).json({ success: false, message: 'Document not found' });

    const canVerify = req.user.role === 'admin';
    if (!canVerify) {
      return res.status(403).json({ success: false, message: 'Unauthorized' });
    }

    document.status = isValid ? 'verified' : 'rejected';
    document.verificationResult = {
      ...document.verificationResult,
      isValid,
      notes: notes || '',
      verifiedBy: req.user.id,
      verifiedAt: new Date(),
      manualReview: true
    };
    await document.save();

    res.json({ success: true, message: `Status updated`, data: document });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Verification failed', error: err.message });
  }
};

const getDocumentOCR = async (req, res) => {
  try {
    const document = await Document.findOne({ _id: req.params.id, userId: req.user.id });
    if (!document) return res.status(404).json({ success: false, message: 'Document not found' });

    if (document.extractedText || document.verificationResult?.analysisDetails?.ocrResult?.text) {
      return res.json({
        success: true,
        data: {
          documentId: document._id,
          extractedText: document.extractedText || document.verificationResult.analysisDetails.ocrResult.text,
          extractedData: document.extractedData || document.verificationResult?.analysisDetails?.ocrResult?.extractedData || {},
          confidence: document.verificationResult?.analysisDetails?.ocrResult?.confidence || 0,
          cached: true
        }
      });
    }

    res.status(400).json({ success: false, message: 'OCR data not available' });

  } catch (error) {
    res.status(500).json({ success: false, message: 'OCR fetch failed', error: error.message });
  }
};

const deleteDocument = async (req, res) => {
  try {
    const document = await Document.findOne({ _id: req.params.id, userId: req.user.id });
    if (!document) return res.status(404).json({ success: false, message: 'Document not found' });

    if (fs.existsSync(document.filePath)) {
      fs.unlinkSync(document.filePath);
    }
    await Document.findByIdAndDelete(req.params.id);
    res.json({ success: true, message: 'Document deleted' });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Delete failed', error: err.message });
  }
};

module.exports = {
  upload,
  uploadDocument,
  getDocuments,
  getDocument,
  getDocumentOCR,
  verifyDocument,
  deleteDocument
};
