import { DBSchema } from 'idb';
import { StoredPrediction } from './types';

export interface PredictionDB extends DBSchema {
  predictions: {
    key: number;
    value: StoredPrediction;
    indexes: { 'by-timestamp': string };
  };
}

export const DB_VERSION = 6;
export const DB_NAME = 'predictions-db';