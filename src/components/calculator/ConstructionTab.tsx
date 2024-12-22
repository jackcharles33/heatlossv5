import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { wallTypes, windowTypes, floorTypes, roofTypes } from '../../constants/buildingData';
import { CalculatorInputs } from '../../types/calculator';

interface ConstructionTabProps {
  inputs: CalculatorInputs;
  onInputChange: (name: string, value: string) => void;
}

export function ConstructionTab({ inputs, onInputChange }: ConstructionTabProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="text-sm font-medium">Wall Construction</label>
        <Select 
          value={inputs.wallType}
          onValueChange={value => onInputChange('wallType', value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select Wall Type" />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(wallTypes).map(([key, { name, uValue }]) => (
              <SelectItem key={key} value={key}>
                <div>
                  <div>{name}</div>
                  <div className="text-sm text-gray-500">U-Value: {uValue}</div>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Windows</label>
        <Select 
          value={inputs.windowType}
          onValueChange={value => onInputChange('windowType', value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select Window Type" />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(windowTypes).map(([key, { name, uValue }]) => (
              <SelectItem key={key} value={key}>
                <div>
                  <div>{name}</div>
                  <div className="text-sm text-gray-500">U-Value: {uValue}</div>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Floor Construction</label>
        <Select 
          value={inputs.floorType}
          onValueChange={value => onInputChange('floorType', value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select Floor Type" />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(floorTypes).map(([key, { name, uValue }]) => (
              <SelectItem key={key} value={key}>
                <div>
                  <div>{name}</div>
                  <div className="text-sm text-gray-500">U-Value: {uValue}</div>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Roof Construction</label>
        <Select 
          value={inputs.roofType}
          onValueChange={value => onInputChange('roofType', value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select Roof Type" />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(roofTypes).map(([key, { name, uValue }]) => (
              <SelectItem key={key} value={key}>
                <div>
                  <div>{name}</div>
                  <div className="text-sm text-gray-500">U-Value: {uValue}</div>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}