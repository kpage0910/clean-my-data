import React, { useState, useRef } from "react";

interface UploadFormProps {
  onUploadSuccess: (fileId: string, filename: string) => void;
  onUploadError: (error: string) => void;
  isLoading?: boolean;
}

export default function UploadForm({
  onUploadSuccess,
  onUploadError,
  isLoading = false,
}: UploadFormProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith(".csv")) {
        setSelectedFile(file);
      } else {
        onUploadError("Please upload a CSV file");
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
    e.target.value = "";
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        process.env.NEXT_PUBLIC_API_URL
          ? `${process.env.NEXT_PUBLIC_API_URL}/upload`
          : "http://localhost:8000/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Upload failed");
      }

      const data = await response.json();
      onUploadSuccess(data.file_id, data.filename);
    } catch (err) {
      onUploadError(err instanceof Error ? err.message : "Upload failed");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all ${
          dragActive
            ? "border-neutral-900 bg-neutral-100"
            : "border-neutral-300 hover:border-neutral-400 bg-white"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          onChange={handleChange}
          className="hidden"
          id="file-upload"
        />

        <div className="space-y-4">
          <div className="w-14 h-14 bg-neutral-100 rounded-2xl flex items-center justify-center mx-auto">
            <svg
              className="w-7 h-7 text-neutral-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>

          <div>
            <label
              htmlFor="file-upload"
              className="cursor-pointer text-neutral-900 hover:text-neutral-600 font-medium"
            >
              Click to upload
            </label>
            <span className="text-neutral-500"> or drag and drop</span>
          </div>

          <p className="text-sm text-neutral-400">CSV files only</p>

          {selectedFile && (
            <div className="mt-4 p-4 bg-neutral-50 rounded-xl border border-neutral-200">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-neutral-200 rounded-lg flex items-center justify-center">
                  <svg
                    className="w-5 h-5 text-neutral-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <div className="text-left">
                  <p className="text-sm font-medium text-neutral-900">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-neutral-500">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <button
        type="submit"
        disabled={!selectedFile || isLoading}
        className={`mt-6 w-full py-3.5 px-4 rounded-full font-medium transition-all ${
          selectedFile && !isLoading
            ? "bg-neutral-900 text-white hover:bg-neutral-800"
            : "bg-neutral-200 text-neutral-400 cursor-not-allowed"
        }`}
      >
        {isLoading ? "Uploading..." : "Upload & Scan"}
      </button>
    </form>
  );
}
