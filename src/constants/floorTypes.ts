export const floorTypes = {
  'concrete-0': { name: 'Concrete Uninsulated', uValue: 0.70 },
  'concrete-25': { name: 'Concrete 25mm insulation', uValue: 0.55 },
  'concrete-50': { name: 'Concrete 50mm insulation', uValue: 0.40 },
  'concrete-75': { name: 'Concrete 75mm insulation', uValue: 0.28 },
  'concrete-100': { name: 'Concrete 100mm insulation', uValue: 0.24 },
  'timber-suspended': { name: 'Timber Suspended Floor', uValue: 0.80 },
  'timber-insulated': { name: 'Timber Insulated Floor', uValue: 0.30 }
} as const;