import { useState } from 'react';
import { Container } from '@mui/material';
import { PropertyAge, PropertyType } from '../types/HouseData';
import { HouseDiagramForm } from './HouseDiagramForm';

interface FormData {
  size: string;
  age: PropertyAge;
  propertyType: PropertyType;
  wallType: string;
  floorType: string;
  windowType: string;
  roofType: string;
}

interface PredictionFormProps {
  onPredict: (data: any) => number;
}

export function PredictionForm({ onPredict }: PredictionFormProps) {
  const [formData, setFormData] = useState<FormData>({
    size: '100',
    age: 'BETWEEN_1960_2000',
    propertyType: 'Semi-Detached / End-Terrace',
    wallType: 'cavity-post60-290-310-filled',
    floorType: 'concrete-75',
    windowType: 'double',
    roofType: 'loft-150'
  });

  const handleChange = (e: any) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict({
      ...formData,
      size: Number(formData.size)
    });
  };

  return (
    <Container maxWidth="lg">
      <HouseDiagramForm 
        values={formData} 
        onChange={handleChange}
        onSubmit={handleSubmit}
      />
    </Container>
  );
}