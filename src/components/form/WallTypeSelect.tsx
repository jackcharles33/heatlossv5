import { FormControl, Select, MenuItem } from '@mui/material';
import { getAvailableWallTypes } from '../../utils/ageCalculations';
import { PropertyAge } from '../../types/HouseData';

interface WallTypeSelectProps {
  value: string;
  onChange: (event: any) => void;
  age: PropertyAge;
}

export function WallTypeSelect({ value, onChange, age }: WallTypeSelectProps) {
  const availableWallTypes = getAvailableWallTypes(age);

  return (
    <FormControl>
      <Select
        name="wallType"
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
        {Object.entries(availableWallTypes).map(([key, { name }]) => (
          <MenuItem key={key} value={key}>{name}</MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}