import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

export const ResultsChart = ({ data }) => {
  const chartData = [
    { name: 'Positive', count: data.positive },
    { name: 'Neutral', count: data.neutral },
    { name: 'Negative', count: data.negative }
  ];

  return (
    <div className="w-full h-64">
      <BarChart
        width={600}
        height={300}
        data={chartData}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="count" fill="#8884d8" />
      </BarChart>
    </div>
  );
};
