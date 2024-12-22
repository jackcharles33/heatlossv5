import { openDB } from 'idb';
import { StoredPrediction } from './types';
import { DB_NAME, DB_VERSION, PredictionDB } from './config';
import { BaseHouseData } from '../types/HouseData';

export const dbPromise = openDB<PredictionDB>(DB_NAME, DB_VERSION, {
  upgrade(db, oldVersion, _newVersion, transaction) {
    if (oldVersion < 1) {
      const store = db.createObjectStore('predictions', {
        keyPath: 'id',
        autoIncrement: true,
      });
      store.createIndex('by-timestamp', 'timestamp');
    }
    
    if (oldVersion < 6) {
      const store = transaction.objectStore('predictions');
      store.openCursor().then(function addConfidence(cursor): Promise<void> | void {
        if (!cursor) return;
        
        const prediction = cursor.value;
        if (!prediction.confidenceScore) {
          prediction.confidenceScore = 0.8;
        }
        
        cursor.update(prediction);
        return cursor.continue().then(addConfidence);
      });
    }
  },
});

export async function clearPredictions(): Promise<void> {
  const db = await dbPromise;
  await db.clear('predictions');
}

export async function storePrediction(
  data: Partial<BaseHouseData>, 
  heatLoss: number, 
  krakenValue?: number,
  confidenceScore: number = 0.8
): Promise<void> {
  const db = await dbPromise;
  
  let accuracy: number | undefined;
  let isAccurate: boolean | undefined;

  if (krakenValue) {
    const percentageDiff = Math.abs((krakenValue - heatLoss) / ((krakenValue + heatLoss) / 2)) * 100;
    accuracy = 100 - percentageDiff;
    isAccurate = percentageDiff <= 10;
  }

  const prediction: StoredPrediction = {
    ...(data as BaseHouseData),
    heatLoss,
    krakenValue,
    accuracy,
    isAccurate,
    confidenceScore,
    timestamp: new Date().toISOString(),
  };

  await db.add('predictions', prediction);
}

export async function getRecentPredictions(limit: number = 100): Promise<StoredPrediction[]> {
  const db = await dbPromise;
  const predictions = await db.getAllFromIndex(
    'predictions',
    'by-timestamp',
    IDBKeyRange.lowerBound(0),
    limit
  );
  return predictions.reverse();
}