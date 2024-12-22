import { FormControl, OutlinedInput } from '@mui/material';
import { FormLabel } from '../styles/FormLabel';

interface FloorAreaInputProps {
  value: string;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

export function FloorAreaInput({ value, onChange }: FloorAreaInputProps) {
  return (
    <FormControl>
      <FormLabel>Floor Area (m²)</FormLabel>
      <OutlinedInput
        name="size"
        type="number"
        value={value}
        onChange={onChange}
        sx={{
          backgroundColor: '#ff4cd4',
          borderRadius: '12px',
          '& input': {
            color: 'rgba(0, 0, 0, 0.87)',
            fontFamily: 'Montserrat, sans-serif'
          }
        }}
      />
    </FormControl>
  );
}