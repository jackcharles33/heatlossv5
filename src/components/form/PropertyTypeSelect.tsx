import { FormControl, Select, MenuItem } from '@mui/material';
import { PropertyType } from '../../types/HouseData';
import { FormLabel } from '../styles/FormLabel';

interface PropertyTypeSelectProps {
  value: PropertyType;
  onChange: (event: any) => void;
}

export function PropertyTypeSelect({ value, onChange }: PropertyTypeSelectProps) {
  return (
    <FormControl>
      <FormLabel>Property Type</FormLabel>
      <Select
        name="propertyType"
        value={value}
        onChange={onChange}
        sx={{
          backgroundColor: '#ff4cd4',
          borderRadius: '12px',
          '& .MuiSelect-select': {
            color: 'rgba(0, 0, 0, 0.87)',
            fontFamily: 'Montserrat, sans-serif'
          }
        }}
      >
        <MenuItem value="Detached">Detached</MenuItem>
        <MenuItem value="Semi-Detached / End-Terrace">Semi-Detached / End-Terrace</MenuItem>
        <MenuItem value="End of Terrace">End of Terrace</MenuItem>
        <MenuItem value="Terrace">Terrace</MenuItem>
        <MenuItem value="Bungalow">Bungalow</MenuItem>
      </Select>
    </FormControl>
  );
}