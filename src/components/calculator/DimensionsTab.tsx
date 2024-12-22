import { Input } from '@/components/ui/input';
import { CalculatorInputs } from '../../types/calculator';

interface DimensionsTabProps {
  inputs: CalculatorInputs;
  onInputChange: (name: string, value: string | number) => void;
}

export function DimensionsTab({ inputs, onInputChange }: DimensionsTabProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="text-sm font-medium">Floor Area (m²)</label>
        <Input 
          type="number"
          placeholder="Enter floor area"
          value={inputs.floorArea}
          onChange={e => onInputChange('floorArea', e.target.value)}
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Number of Stories</label>
        <Input 
          type="number"
          placeholder="Enter number of stories"
          value={inputs.stories}
          onChange={e => onInputChange('stories', parseInt(e.target.value) || 1)}
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Indoor Temperature (°C)</label>
        <Input 
          type="number"
          placeholder="Default: 20°C"
          value={inputs.indoorTemp}
          onChange={e => onInputChange('indoorTemp', parseInt(e.target.value) || 20)}
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Glazing Ratio (%)</label>
        <Input 
          type="number"
          placeholder="Default: 20%"
          value={inputs.glazingRatio}
          onChange={e => onInputChange('glazingRatio', parseInt(e.target.value) || 20)}
        />
      </div>
    </div>
  );
}