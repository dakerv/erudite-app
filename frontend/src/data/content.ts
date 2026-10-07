import type { Assessment } from '../types/detection';

export const FEATURED_IMAGE = "/3f810dc8-de92-45b9-a94b-5556f5e4b27e.jpg";


export interface ImageType {
  index: string;
  label: string;
  caption: string;
  image: string;
  credit: string;
}

export const imageTypes: ImageType[] = [
{
  index: '01',
  label: 'REAL',
  caption:
  'An image captured without synthetic generation or face replacement.',
  image: "/00068.jpg",

  credit: 'ARCHIVE / 35MM · UNALTERED CAPTURE'
},
{
  index: '02',
  label: 'FACE-SWAPPED',
  caption:
  'An existing image in which facial content has been digitally replaced or manipulated.',
  image: "/45800_45799.jpg",

  credit: 'ARCHIVE / 35MM · FACIAL REGION REPLACED'
},
{
  index: '03',
  label: 'SYNTHETIC',
  caption:
  'An image generated artificially rather than captured as a conventional photograph.',
  image: "/fake_25.jpg",

  credit: 'NO CAMERA ORIGIN · FULLY GENERATED'
}];


export interface WorkStep {
  index: string;
  title: string;
  body: string;
}

export const workSteps: WorkStep[] = [
{
  index: '01',
  title: 'UPLOAD',
  body: 'Select an image from your device.'
},
{
  index: '02',
  title: 'ANALYZE',
  body: 'The application processes the image using the detection model.'
},
{
  index: '03',
  title: 'CLASSIFY',
  body: 'The model evaluates the image across three classes.'
},
{
  index: '04',
  title: 'ASSESS',
  body:
  'The application presents the predicted class and confidence distribution.'
}];


export interface Topic {
  index: string;
  title: string;
  body: string;
  image: string;
  caption: string;
}

export const topics: Topic[] = [
{
  index: 'I',
  title: 'MISINFORMATION',
  body:
  'Synthetic images can make fabricated events appear genuine and influence how people understand real-world events.',
  image: "/e8801c4e-c526-43ed-a182-e621a2ff366c.jpg",

  caption: 'Circulation outpaces correction.'
},
{
  index: 'II',
  title: 'IDENTITY ABUSE',
  body:
  'Face manipulation can make people appear in situations they were never part of.',
  image: "/61bdb98e-ba99-4e8f-9a2d-25785e28dc5b.jpg",

  caption: 'A likeness, borrowed without consent.'
},
{
  index: 'III',
  title: 'LOSS OF TRUST',
  body:
  'As synthetic media becomes increasingly convincing, determining what visual information can be trusted becomes more difficult.',
  image: "/8bf730f0-b7e7-4803-b197-27d23816425d.jpg",

  caption: 'Doubt becomes the default reading.'
}];


export const sampleAssessment: Assessment = {
  id: '001',
  fileName: 'corridor_frame_04.jpg',
  imageUrl: "/ad7c7ebc-689b-42a8-a4e6-5694f6606811.jpg",

  predicted: 'swapped',
  confidence: 93.6,
  scores: [
  { id: 'real', label: 'REAL', probability: 4.8 },
  { id: 'swapped', label: 'FACE-SWAPPED', probability: 93.6 },
  { id: 'synthetic', label: 'SYNTHETIC', probability: 1.6 }]

};