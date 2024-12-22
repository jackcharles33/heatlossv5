interface ElementHeatLossInputs {
  netWallArea: number;
  glazingArea: number;
  floorArea: number;
  roofArea: number;
  wallUValue: number;
  windowUValue: number;
  floorUValue: number;
  roofUValue: number;
  tempDiff: number;
}

interface ElementHeatLoss {
  wallLoss: number;
  windowLoss: number;
  floorLoss: number;
  roofLoss: number;
}

export function calculateElementHeatLoss(inputs: ElementHeatLossInputs): ElementHeatLoss {
  return {
    wallLoss: inputs.netWallArea * inputs.wallUValue * inputs.tempDiff,
    windowLoss: inputs.glazingArea * inputs.windowUValue * inputs.tempDiff,
    floorLoss: inputs.floorArea * inputs.floorUValue * inputs.tempDiff,
    roofLoss: inputs.roofArea * inputs.roofUValue * inputs.tempDiff
  };
}