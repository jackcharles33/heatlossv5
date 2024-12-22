import { FormControl, Select, MenuItem } from '@mui/material';
import { windowTypes } from '../../constants/buildingData';

interface WindowTypeSelectProps {
  value: string;
  onChange: (event: any) => void;
}

export function WindowTypeSelect({ value, onChange }: WindowTypeSelectProps) {
  return (
    <FormControl>
      <Select
        name="windowType"
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
        {Object.entries(windowTypes).map(([key, { name }]) => (
          <MenuItem key={key} value={key}>{name}</MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}