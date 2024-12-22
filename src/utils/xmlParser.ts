import { XMLParser } from 'fast-xml-parser';
import { BaseHouseData, PropertyAge, PropertyType, WindowType } from '../types/HouseData';

interface XMLHouseData {
  size: string;
  age: PropertyAge;
  propertyType: PropertyType;
  windowType?: WindowType;
  wallType?: string;
  floorType?: string;
  roofType?: string;
  heatLoss: string;
}

interface XMLResponse {
  houses: {
    house: XMLHouseData | XMLHouseData[];
  };
}

export function parseXMLData(xmlContent: string): BaseHouseData[] {
  const parser = new XMLParser();
  const jsonObj = parser.parse(xmlContent) as XMLResponse;
  
  const houses = jsonObj.houses.house;
  const houseArray = Array.isArray(houses) ? houses : [houses];
  
  return houseArray.map((house): BaseHouseData => ({
    size: Number(house.size),
    age: house.age,
    propertyType: house.propertyType,
    windowType: house.windowType || 'double',
    wallType: house.wallType || 'cavity-post60-290-310-filled',
    floorType: house.floorType || 'concrete-75',
    roofType: house.roofType || 'loft-150',
    heatLoss: Number(house.heatLoss)
  }));
}