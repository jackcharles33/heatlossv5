import { BaseHouseData } from '../types/HouseData';
import { parseXMLData } from './xmlParser';
import { parseCSVData } from './csvParser';
import { trainingData } from '../data/trainingData';

export function parseFileContent(content: string): BaseHouseData[] {
  if (content.trim().startsWith('<?xml') || content.includes('<houses>')) {
    return parseXMLData(content);
  }
  
  // Parse CSV string into array of objects
  const lines = content.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  
  const csvData = lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim());
    return headers.reduce((obj: any, header, index) => {
      obj[header] = values[index];
      return obj;
    }, {});
  });
  
  return parseCSVData(csvData);
}

export function getInitialTrainingData(): BaseHouseData[] {
  return parseCSVData(trainingData);
}