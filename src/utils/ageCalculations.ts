import { PropertyAge } from '../types/HouseData';
import { wallTypes } from '../constants/buildingData';

export const ageMultipliers: Record<PropertyAge, number> = {
  'PRE_1960': 0.16,      // Decreased by 20% from 0.2
  'BETWEEN_1960_2000': 0.528,  // Decreased by 20% from 0.66
  'BETWEEN_2000_2008': 0.496,  // Decreased by 20% from 0.62
  'POST_2008': 0.144     // Decreased by 20% from 0.18
};

export function getAvailableWallTypes(age: PropertyAge) {
  switch (age) {
    case 'PRE_1960':
      return {
        'solid-brick-102': wallTypes['solid-brick-102'],
        'solid-brick-228': wallTypes['solid-brick-228'],
        'solid-brick-343': wallTypes['solid-brick-343'],
        'cavity-pre60-unfilled': wallTypes['cavity-pre60-unfilled'],
        'cavity-pre60-filled': wallTypes['cavity-pre60-filled']
      };
    case 'BETWEEN_1960_2000':
      return {
        'cavity-post60-290-310-filled': wallTypes['cavity-post60-290-310-filled'],
        'cavity-post60-290-310-unfilled': wallTypes['cavity-post60-290-310-unfilled'],
        'cavity-post60-under290-filled': wallTypes['cavity-post60-under290-filled'],
        'cavity-post60-under290-unfilled': wallTypes['cavity-post60-under290-unfilled']
      };
    case 'BETWEEN_2000_2008':
    case 'POST_2008':
      return {
        'cavity-post60-310': wallTypes['cavity-post60-310'],
        'timber-frame': wallTypes['timber-frame']
      };
    default:
      return wallTypes;
  }
}