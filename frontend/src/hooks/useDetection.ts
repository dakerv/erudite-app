import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  Assessment,
  DetectionStage,
  DetectionStatus
} from '../types/detection';


// ==================================
// ACCEPTED FILE TYPES AND SIZE LIMIT
// ==================================

// Image formats accepted by the detection system.
const ACCEPTED = [
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/webp'
];

// Maximum image size: 12 MB.
const MAX_BYTES = 12 * 1024 * 1024;


// ========================
// DETECTION STAGE SEQUENCE
// ========================

// Stages displayed to the user while an image is being
// processed.
//
// The "at" value specifies when each stage appears,
// measured in milliseconds after analysis begins.
//
// These stages are for frontend feedback. The actual
// preprocessing and model inference happen in Flask.
const STAGE_SEQUENCE: {
  stage: DetectionStage;
  label: string;
  at: number;
}[] = [
  {
    stage: 'received',
    label: 'IMAGE RECEIVED',
    at: 0
  },
  {
    stage: 'preprocessing',
    label: 'PREPROCESSING',
    at: 700
  },
  {
    stage: 'analyzing',
    label: 'ANALYZING',
    at: 1600
  },
  {
    stage: 'generating',
    label: 'GENERATING ASSESSMENT',
    at: 2600
  }
];


// Make the stage information available to other components.
export const STAGE_LABELS = STAGE_SEQUENCE.map((s) => ({
  stage: s.stage,
  label: s.label
}));


// =======================
// FLASK API RESPONSE TYPE
// =======================

// Structure of the JSON response returned by Flask.
//
// Flask returns confidence and probabilities as decimal
// values between 0 and 1.
//
// Example:
//
//     confidence: 0.94
//
// The frontend converts this to:
//
//     94.0%
interface PredictionResponse {
  prediction: 'real' | 'synthetic' | 'swapped';

  confidence: number;

  probabilities: {
    real: number;
    synthetic: number;
    swapped: number;
  };

  // Flask includes this when an error occurs.
  error?: string;
}


// ==============
// DETECTION HOOK
// ==============

// This hook manages the complete image detection workflow:
//
//     Select image
//          ↓
//     Validate image
//          ↓
//     Preview image
//          ↓
//     Send image to Flask
//          ↓
//     EfficientNet-B0 processes image
//          ↓
//     Receive prediction and probabilities
//          ↓
//     Display assessment
export function useDetection() {


  // =================
  // APPLICATION STATE
  // =================

  // Current detection status.
  const [status, setStatus] =
    useState<DetectionStatus>('empty');

  // Current processing stage.
  const [stage, setStage] =
    useState<DetectionStage>('received');

  // Image selected by the user.
  const [file, setFile] =
    useState<File | null>(null);

  // Temporary browser URL used to display the image.
  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null);

  // Final assessment returned by the model.
  const [assessment, setAssessment] =
    useState<Assessment | null>(null);

  // Error message shown to the user.
  const [error, setError] =
    useState<string | null>(null);


  // ==================
  // ASSESSMENT COUNTER
  // ==================

  // Gives each completed assessment a simple ID:
  //
  //     001
  //     002
  //     003
  //
  // useRef is used so changing the counter does not
  // cause the component to re-render.
  const counter = useRef(1);


  // ================
  // TIMER MANAGEMENT
  // =================

  // Stores active processing-stage timers.
  const timers = useRef<number[]>([]);


  // Cancel all currently active timers.
  const clearTimers = useCallback(() => {

    timers.current.forEach((t) => {
      window.clearTimeout(t);
    });

    timers.current = [];
  }, []);


  // Clean up timers when the hook is removed.
  useEffect(() => clearTimers, [clearTimers]);


  // ==========================================================
  // RESET DETECTION
  // ==========================================================

  // Returns the application to its initial state.
  const reset = useCallback(() => {

    clearTimers();

    setStatus('empty');
    setStage('received');
    setFile(null);

    // Release the temporary browser URL.
    setPreviewUrl((url) => {

      if (url) {
        URL.revokeObjectURL(url);
      }

      return null;
    });

    setAssessment(null);
    setError(null);

  }, [clearTimers]);


  // ==========================================================
  // FILE SELECTION AND VALIDATION
  // ==========================================================

  // Called when the user selects an image.
  //
  // Performs:
  //     1. File existence check
  //     2. File type validation
  //     3. File size validation
  //     4. Preview creation
  const selectFile = useCallback(
    (next: File | null | undefined) => {

      clearTimers();

      // Remove any previous assessment.
      setAssessment(null);


      // --------------------------------------------------------
      // NO FILE
      // --------------------------------------------------------

      if (!next) {
        return;
      }


      // --------------------------------------------------------
      // FILE TYPE VALIDATION
      // --------------------------------------------------------

      if (!ACCEPTED.includes(next.type.toLowerCase())) {

        setFile(null);
        setPreviewUrl(null);

        setError(
          `“${next.name}” is not a supported format. This examiner accepts JPG, PNG and WEBP images only.`
        );

        setStatus('error');

        return;
      }


      // --------------------------------------------------------
      // FILE SIZE VALIDATION
      // --------------------------------------------------------

      if (next.size > MAX_BYTES) {

        setFile(null);
        setPreviewUrl(null);

        setError(
          `“${next.name}” exceeds the 12 MB limit. Submit a smaller version of the image.`
        );

        setStatus('error');

        return;
      }


      // --------------------------------------------------------
      // ACCEPT VALID FILE
      // --------------------------------------------------------

      setError(null);
      setFile(next);


      // Create temporary browser URL for image preview.
      const url = URL.createObjectURL(next);


      // Replace the previous preview URL.
      setPreviewUrl((prev) => {

        if (prev) {
          URL.revokeObjectURL(prev);
        }

        return url;
      });


      // Image is ready to be analyzed.
      setStatus('ready');

    },
    [clearTimers]
  );


  // ==========================================================
  // SEND IMAGE TO FLASK / RUN MODEL
  // ==========================================================

  // This function replaces the previous dummy detection system.
  //
  // The previous version generated a fake prediction locally.
  //
  // The new workflow is:
  //
  //     React
  //       ↓
  //     FormData
  //       ↓
  //     POST /predict
  //       ↓
  //     Flask
  //       ↓
  //     EfficientNet-B0
  //       ↓
  //     Prediction + probabilities
  const analyze = useCallback(async () => {

    // Do nothing if there is no image to analyze.
    if (!file || !previewUrl) {
      return;
    }


    // --------------------------------------------------------
    // START ANALYSIS
    // --------------------------------------------------------

    clearTimers();

    setError(null);
    setStatus('analyzing');
    setStage('received');


    // --------------------------------------------------------
    // DISPLAY PROCESSING STAGES
    // --------------------------------------------------------

    // These timers control the visual progress indicator.
    //
    // They do not perform the actual model processing.
    STAGE_SEQUENCE.forEach(({ stage: s, at }) => {

      if (at === 0) {
        return;
      }

      timers.current.push(
        window.setTimeout(() => {
          setStage(s);
        }, at)
      );

    });


    // --------------------------------------------------------
    // PREPARE IMAGE FOR REQUEST
    // --------------------------------------------------------

    try {

      // FormData allows the image to be sent as
      // multipart/form-data.
      //
      // Flask receives this using:
      //
      //     request.files["image"]
      const formData = new FormData();

      formData.append('image', file);


      // ======================================================
      // SEND IMAGE TO FLASK
      // ======================================================

      // /predict is the Flask endpoint responsible for
      // running the trained EfficientNet-B0 model.
      //
      // We do not manually set Content-Type because the
      // browser automatically handles the multipart boundary
      // when FormData is used.

      const API_URL =
        import.meta.env.DEV
          ? ''
          : 'https://erudite-app-backend.onrender.com';

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
      });


      // Convert Flask JSON response into a TypeScript object.
      const data: PredictionResponse =
        await response.json();


      // ======================================================
      // HANDLE BACKEND ERRORS
      // ======================================================

      if (!response.ok) {

        throw new Error(
          data.error ||
          'The image could not be analyzed.'
        );
      }


      // ======================================================
      // CREATE ASSESSMENT ID
      // ======================================================

      const id =
        String(counter.current).padStart(3, '0');

      counter.current += 1;


      // ======================================================
      // MODEL CLASS ORDER
      // ======================================================

      // This order matches the classes used by the trained model:
      //
      //     0 → real
      //     1 → synthetic
      //     2 → swapped
      const order: (
        'real' |
        'synthetic' |
        'swapped'
      )[] = [
        'real',
        'synthetic',
        'swapped'
      ];


      // ======================================================
      // CONVERT MODEL RESPONSE TO FRONTEND FORMAT
      // ======================================================

      const assessmentResult: Assessment = {

        id,

        // Original filename.
        fileName: file.name,

        // Image displayed in the result.
        imageUrl: previewUrl,

        // Class selected by EfficientNet-B0.
        predicted: data.prediction,

        // Flask returns a decimal probability.
        //
        // Example:
        //
        //     0.999002
        //
        // The UI should display:
        //
        //     99.9%
        //
        // Therefore multiply by 100 and round to one
        // decimal place.
        confidence:
          Number(
            (data.confidence * 100).toFixed(1)
          ),

        // Convert all three probabilities to percentages.
        scores: order.map((className) => ({

          id: className,

          // Display the exact model class names.
          //
          // REAL
          // SYNTHETIC
          // SWAPPED
          label: className.toUpperCase(),

          probability:
            Number(
              (
                data.probabilities[className] * 100
              ).toFixed(1)
            )

        }))
      };


      // ======================================================
      // STORE MODEL RESULT
      // ======================================================

      setAssessment(assessmentResult);

      setStage('generating');

      setStatus('result');


    } catch (err) {

      // Log technical error information for development.
      console.error(
        'Detection request failed:',
        err
      );


      // Stop processing timers.
      clearTimers();


      // Display a user-friendly error.
      setError(
        err instanceof Error
          ? err.message
          : 'An unexpected error occurred while analyzing the image.'
      );


      // Put application into error state.
      setStatus('error');
    }

  }, [file, previewUrl, clearTimers]);


  // ==========================================================
  // RETURN STATE AND FUNCTIONS
  // ==========================================================

  // These values are consumed by components such as
  // UploadPanel, Home and AssessmentPage.
  return {
    status,
    stage,
    file,
    previewUrl,
    assessment,
    error,
    selectFile,
    analyze,
    reset
  };
}