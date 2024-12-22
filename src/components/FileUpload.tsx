import React from 'react';
import { Button } from '@mui/material';

interface FileUploadProps {
  onFileUpload: (data: string) => void;
  accept?: string;
}

export function FileUpload({ onFileUpload, accept = ".xml,.csv" }: FileUploadProps) {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        onFileUpload(content);
      };
      reader.readAsText(file);
    }
  };

  return (
    <Button variant="contained" component="label">
      Upload Training Data
      <input
        type="file"
        accept={accept}
        hidden
        onChange={handleFileChange}
      />
    </Button>
  );
}