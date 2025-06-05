import React from 'react';
import { Upload } from 'lucide-react';

export const FileUpload = ({ onFileSelect, loading }) => {
  return (
    <div className="flex items-center justify-center w-full">
      <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
        <div className="flex flex-col items-center justify-center pt-5 pb-6">
          <Upload className="w-8 h-8 mb-4 text-gray-500" />
          <p className="mb-2 text-sm text-gray-500">
            <span className="font-semibold">Click to upload</span> or drag and drop
          </p>
          <p className="text-xs text-gray-500">CSV files only</p>
        </div>
        <input
          type="file"
          className="hidden"
          accept=".csv"
          onChange={onFileSelect}
          disabled={loading}
        />
      </label>
    </div>
  );
};
