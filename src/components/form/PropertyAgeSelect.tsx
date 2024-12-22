import { FormControl, Select, MenuItem } from '@mui/material';
import { PropertyAge } from '../../types/HouseData';
import { FormLabel } from '../styles/FormLabel';

interface PropertyAgeSelectProps {
  value: PropertyAge;
  onChange: (event: any) => void;
}

export function PropertyAgeSelect({ value, onChange }: PropertyAgeSelectProps) {
  return (
    <FormControl>
      <FormLabel>Property Age</FormLabel>
      <Select
        name="age"
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
        <MenuItem value="PRE_1960">Pre 1960</MenuItem>
        <MenuItem value="BETWEEN_1960_2000">1960-2000</MenuItem>
        <MenuItem value="BETWEEN_2000_2008">2000-2008</MenuItem>
        <MenuItem value="POST_2008">Post 2008</MenuItem>
      </Select>
    </FormControl>
  );
}