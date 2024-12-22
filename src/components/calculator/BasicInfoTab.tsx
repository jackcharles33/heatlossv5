import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { UKRegions } from '../../constants/buildingData';
import { CalculatorInputs } from '../../types/calculator';

interface BasicInfoTabProps {
  inputs: CalculatorInputs;
  onInputChange: (name: string, value: string) => void;
}

export function BasicInfoTab({ inputs, onInputChange }: BasicInfoTabProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="text-sm font-medium">Region</label>
        <Select 
          value={inputs.region} 
          onValueChange={value => onInputChange('region', value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select Region" />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(UKRegions).map(([key, { name }]) => (
              <SelectItem key={key} value={key}>{name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Postcode</label>
        <Input 
          placeholder="Enter postcode"
          value={inputs.postcode}
          onChange={e => onInputChange('postcode', e.target.value)}
        />
      </div>
    </div>
  );
}