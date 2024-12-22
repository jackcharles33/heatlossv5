import { FormControl, Select, MenuItem } from '@mui/material';
import { roofTypes } from '../../constants/buildingData';

interface RoofTypeSelectProps {
  value: string;
  onChange: (event: any) => void;
}

export function RoofTypeSelect({ value, onChange }: RoofTypeSelectProps) {
  return (
    <FormControl>
      <Select
        name="roofType"
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
        {Object.entries(roofTypes).map(([key, { name }]) => (
          <MenuItem key={key} value={key}>{name}</MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}