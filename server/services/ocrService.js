const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

class OCRService {
  constructor() {
    // OCR.space API configuration - use environment variable for API key
    const dotenv = require('dotenv');
    dotenv.config();
    
    this.apiKey = process.env.OCR_SPACE_API_KEY || 'K87898142388957'; // Fallback free tier key
    this.baseUrl = 'https://api.ocr.space/parse/image';
    this.defaultOptions = {
      language: 'eng',
      isOverlayRequired: false,
      detectOrientation: true,
      isTable: false,
      scale: true,
      OCREngine: 2, // Use OCR Engine 2 for better accuracy
      isCreateSearchablePdf: false,
      isSearchablePdfHideTextLayer: false
    };
    
    if (!process.env.OCR_SPACE_API_KEY) {
      console.warn('[OCRService] Warning: OCR_SPACE_API_KEY not set in environment. Using free tier (rate limited)');
    }
  }

  /**
   * Extract text from image/PDF using OCR.space API
   * @param {string} filePath - Path to the file
   * @param {Object} options - OCR options
   * @returns {Object} OCR result with extracted text and confidence
   */
  async extractText(filePath, options = {}) {
    try {
      console.log(`[OCRService] Starting OCR extraction for: ${filePath}`);
      
      // Check if file exists
      if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
      }

      // Check file size (OCR.space has limits)
      const stats = fs.statSync(filePath);
      const fileSizeInMB = stats.size / (1024 * 1024);
      if (fileSizeInMB > 10) {
        throw new Error(`File size too large: ${fileSizeInMB.toFixed(2)}MB. Maximum allowed: 10MB`);
      }

      // Prepare form data
      const formData = new FormData();
      
      // Read file as buffer to avoid stream issues
      const fileBuffer = fs.readFileSync(filePath);
      const fileName = path.basename(filePath);
      
      // Add file with proper options
      formData.append('file', fileBuffer, {
        filename: fileName,
        contentType: this.getContentType(filePath)
      });
      
      // Add API key
      formData.append('apikey', this.apiKey);
      
      // Add OCR options
      const ocrOptions = { ...this.defaultOptions, ...options };
      Object.keys(ocrOptions).forEach(key => {
        formData.append(key, String(ocrOptions[key]));
      });

      console.log(`[OCRService] Making OCR API request for file: ${fileName} (${fileSizeInMB.toFixed(2)}MB)`);

      // Make API request
      const response = await axios.post(this.baseUrl, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 30000, // 30 seconds timeout
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      });

      console.log(`[OCRService] OCR API Response Status: ${response.status}`);
      
      if (response.data.IsErroredOnProcessing) {
        const errorMessages = response.data.ErrorMessage || response.data.ErrorDetails || ['Unknown error'];
        const errorMsg = Array.isArray(errorMessages) ? errorMessages.join(', ') : errorMessages;
        throw new Error(`OCR processing error: ${errorMsg}`);
      }

      // Extract and process results
      const result = this.processOCRResult(response.data);
      console.log(`[OCRService] OCR extraction completed. Text length: ${result.text.length}`);
      
      return result;

    } catch (error) {
      console.error(`[OCRService] OCR extraction failed:`, error.message);
      
      // Return fallback result
      return {
        success: false,
        text: '',
        confidence: 0,
        error: error.message,
        extractedData: {},
        textRegions: [],
        detectedLanguage: 'unknown'
      };
    }
  }

  /**
   * Process OCR API response and extract useful information
   * @param {Object} ocrResponse - Raw OCR API response
   * @returns {Object} Processed OCR result
   */
  processOCRResult(ocrResponse) {
    try {
      const parsedResults = ocrResponse.ParsedResults || [];
      
      if (parsedResults.length === 0) {
        return {
          success: false,
          text: '',
          confidence: 0,
          extractedData: {},
          textRegions: [],
          detectedLanguage: 'unknown'
        };
      }

      // Get the first parsed result
      const mainResult = parsedResults[0];
      const extractedText = mainResult.ParsedText || '';
      
      // Calculate confidence (OCR.space doesn't provide confidence, so we estimate it)
      const confidence = this.estimateConfidence(extractedText, mainResult);
      
      // Extract structured data from text
      const extractedData = this.extractStructuredData(extractedText);
      
      // Get text regions if available
      const textRegions = this.extractTextRegions(mainResult);

      return {
        success: true,
        text: extractedText.trim(),
        confidence: confidence,
        extractedData: extractedData,
        textRegions: textRegions,
        detectedLanguage: mainResult.Language || 'eng',
        processingTime: ocrResponse.ProcessingTimeInMilliseconds || 0,
        ocrEngine: mainResult.OCREngine || 'unknown'
      };

    } catch (error) {
      console.error(`[OCRService] Error processing OCR result:`, error.message);
      return {
        success: false,
        text: '',
        confidence: 0,
        error: error.message,
        extractedData: {},
        textRegions: [],
        detectedLanguage: 'unknown'
      };
    }
  }

  /**
   * Estimate confidence based on text quality and length - STRICTER VERSION
   * @param {string} text - Extracted text
   * @param {Object} result - OCR result object
   * @returns {number} Confidence score (0-1)
   */
  estimateConfidence(text, result) {
    if (!text || text.length === 0) return 0;

    // Start with much lower base confidence - be skeptical!
    let confidence = 0.2; // Much more conservative starting point

    // Text length factor - stricter requirements
    if (text.length > 50) confidence += 0.1;   // Require more text for bonus
    if (text.length > 150) confidence += 0.1;  // Higher threshold
    if (text.length > 400) confidence += 0.1;  // Much higher threshold

    // Check for ESSENTIAL document patterns - much stricter
    const criticalPatterns = {
      dates: /\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{2}[-/]\d{2}[-/]\d{4}\b/, // Dates
      ids: /\b[A-Z]{2,3}\d{6,}\b|\b\d{6,}\b/, // ID numbers (6+ digits)
      names: /\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b/, // Full names (first + last)
      officialKeywords: /\b(LICENSE|PASSPORT|CERTIFICATE|IDENTIFICATION|IDENTITY|CARD|GOVERNMENT|ISSUED|OFFICIAL|DOCUMENT)\b/i,
      addresses: /\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b/i,
      stateCode: /\b[A-Z]{2}\s+\d{5}\b/, // State + ZIP
      documentNumbers: /\b[A-Z0-9]{8,}\b/, // Long alphanumeric codes
      phoneNumbers: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/
    };

    let criticalMatches = 0;
    let totalCriticalPatterns = Object.keys(criticalPatterns).length;

    // Check each critical pattern
    for (const [patternName, pattern] of Object.entries(criticalPatterns)) {
      if (pattern.test(text)) {
        criticalMatches++;
        confidence += 0.05; // Smaller individual boosts
        console.log(`[OCR] Found critical pattern: ${patternName}`);
      } else {
        console.log(`[OCR] Missing critical pattern: ${patternName}`);
      }
    }

    // Penalty for missing too many critical patterns
    const missingCriticalRatio = (totalCriticalPatterns - criticalMatches) / totalCriticalPatterns;
    if (missingCriticalRatio > 0.6) {
      confidence *= 0.5; // Heavy penalty for missing most critical patterns
      console.log(`[OCR] Missing ${missingCriticalRatio * 100}% of critical patterns - applying penalty`);
    }

    // Require at least 3 critical patterns for decent confidence
    if (criticalMatches < 3) {
      confidence *= 0.6;
      console.log(`[OCR] Only found ${criticalMatches} critical patterns - applying penalty`);
    }

    // Check for meaningful words (not just random characters)
    const words = text.split(/\s+/).filter(word => word.length > 1);
    const meaningfulWords = words.filter(word => /^[A-Za-z0-9]+$/.test(word));
    const meaningfulRatio = meaningfulWords.length / Math.max(words.length, 1);
    
    if (meaningfulRatio < 0.7) {
      confidence *= 0.7; // Penalty for too many garbled words
      console.log(`[OCR] Low meaningful word ratio: ${meaningfulRatio} - applying penalty`);
    }

    // Check for proper document structure indicators
    const structureIndicators = [
      /\b(NAME|FULL NAME|FIRST NAME|LAST NAME)[:]\s*[A-Z]/i,
      /\b(DATE OF BIRTH|DOB|BIRTH DATE)[:]\s*\d/i,
      /\b(ADDRESS|HOME ADDRESS)[:]\s*\d/i,
      /\b(ID NUMBER|LICENSE NUMBER|DOCUMENT NUMBER)[:]\s*[A-Z0-9]/i,
      /\b(ISSUED|EXPIRES|EXPIRATION)[:]\s*\d/i,
      /\b(SEX|GENDER)[:]\s*(M|F|MALE|FEMALE)/i
    ];

    let structureMatches = 0;
    structureIndicators.forEach(pattern => {
      if (pattern.test(text)) {
        structureMatches++;
        confidence += 0.08; // Good bonus for structured data
      }
    });

    console.log(`[OCR] Found ${structureMatches} structure indicators`);

    // STRONG penalty for very short text (likely fake or poor quality)
    if (text.length < 30) {
      confidence *= 0.2;
      console.log(`[OCR] Very short text (${text.length} chars) - applying heavy penalty`);
    } else if (text.length < 80) {
      confidence *= 0.5;
      console.log(`[OCR] Short text (${text.length} chars) - applying penalty`);
    }

    // Check for fake/test document indicators
    const fakeIndicators = [
      /\b(FAKE|TEST|SAMPLE|DUMMY|PLACEHOLDER|EXAMPLE|DEMO)\b/i,
      /\b(JOHN DOE|JANE DOE|TEST USER|SAMPLE USER)\b/i,
      /\b(123-45-6789|000-00-0000)\b/, // Common fake SSNs
      /\b(123 MAIN ST|123 FAKE ST)\b/i // Common fake addresses
    ];

    fakeIndicators.forEach(pattern => {
      if (pattern.test(text)) {
        confidence *= 0.1; // Heavy penalty for fake indicators
        console.log(`[OCR] Detected fake document indicator - applying heavy penalty`);
      }
    });

    // Final confidence bounds and logging
    const finalConfidence = Math.max(0.0, Math.min(0.95, confidence));
    
    console.log(`[OCR] Confidence calculation:`, {
      textLength: text.length,
      criticalMatches: `${criticalMatches}/${totalCriticalPatterns}`,
      structureMatches,
      meaningfulRatio: meaningfulRatio.toFixed(2),
      finalConfidence: finalConfidence.toFixed(2)
    });

    return finalConfidence;
  }

  /**
   * Extract structured data from text based on common document patterns
   * @param {string} text - Extracted text
   * @returns {Object} Structured data
   */
  extractStructuredData(text) {
    const data = {};

    try {
      // Common patterns for different document types
      const patterns = {
        // Dates
        dates: /\b(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b/g,
        
        // Names (basic pattern)
        names: /\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b/g,
        
        // Document numbers
        documentNumbers: /\b([A-Z0-9]{6,})\b/g,
        
        // Phone numbers
        phoneNumbers: /\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b/g,
        
        // Email addresses
        emails: /\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b/g,
        
        // Addresses (basic pattern)
        addresses: /\b(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl)\b)/gi
      };

      Object.keys(patterns).forEach(key => {
        const matches = text.match(patterns[key]);
        if (matches) {
          data[key] = matches;
        }
      });

      // Extract specific document fields based on keywords
      const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
      
      lines.forEach(line => {
        // Look for common document field patterns
        if (line.toLowerCase().includes('name') && line.includes(':')) {
          const nameMatch = line.split(':')[1]?.trim();
          if (nameMatch) data.fullName = nameMatch;
        }
        
        if (line.toLowerCase().includes('date of birth') || line.toLowerCase().includes('dob')) {
          const dobMatch = line.match(/\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b/);
          if (dobMatch) data.dateOfBirth = dobMatch[0];
        }
        
        if (line.toLowerCase().includes('document number') || line.toLowerCase().includes('id number')) {
          const numMatch = line.match(/\b[A-Z0-9]{6,}\b/);
          if (numMatch) data.documentNumber = numMatch[0];
        }
      });

    } catch (error) {
      console.error(`[OCRService] Error extracting structured data:`, error.message);
    }

    return data;
  }

  /**
   * Extract text regions from OCR result
   * @param {Object} result - OCR result object
   * @returns {Array} Array of text regions with coordinates
   */
  extractTextRegions(result) {
    const regions = [];

    try {
      // OCR.space doesn't provide detailed word-level coordinates by default
      // This is a placeholder for text regions
      if (result.ParsedText) {
        const lines = result.ParsedText.split('\n').filter(line => line.trim().length > 0);
        
        lines.forEach((line, index) => {
          regions.push({
            text: line.trim(),
            confidence: 0.8, // Estimated confidence
            line: index + 1,
            // Note: OCR.space would need additional configuration to get actual coordinates
            coordinates: null
          });
        });
      }
    } catch (error) {
      console.error(`[OCRService] Error extracting text regions:`, error.message);
    }

    return regions;
  }

  /**
   * Get content type based on file extension
   * @param {string} filePath - Path to the file
   * @returns {string} Content type
   */
  getContentType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const contentTypes = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.bmp': 'image/bmp',
      '.tiff': 'image/tiff',
      '.tif': 'image/tiff',
      '.webp': 'image/webp',
      '.pdf': 'application/pdf'
    };
    return contentTypes[ext] || 'application/octet-stream';
  }

  /**
   * Get OCR service health status
   * @returns {Object} Health status
   */
  async getHealthStatus() {
    try {
      // Simple health check by making a minimal API call
      const testResponse = await axios.get('https://api.ocr.space/', {
        timeout: 5000
      });
      
      return {
        status: 'healthy',
        apiKey: this.apiKey ? 'configured' : 'missing',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message,
        apiKey: this.apiKey ? 'configured' : 'missing',
        timestamp: new Date().toISOString()
      };
    }
  }
}

module.exports = new OCRService();
