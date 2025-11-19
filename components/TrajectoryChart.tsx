import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrajectoryPoint } from '../types';

interface TrajectoryChartProps {
  data: TrajectoryPoint[];
  isAi: boolean;
}

const TrajectoryChart: React.FC<TrajectoryChartProps> = ({ data, isAi }) => {
  const color = isAi ? '#f43f5e' : '#34d399'; // Red for AI, Green for Real

  return (
    <div className="w-full h-64 bg-slate-900/50 rounded-lg border border-slate-800 p-4 relative overflow-hidden">
      <div className="absolute top-4 left-4 z-10">
        <h3 className="text-sm font-semibold text-slate-200">Latent Space Trajectory</h3>
        <p className="text-xs text-slate-500">Visualizing temporal curvature (DINOv2 Space)</p>
      </div>
      
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis type="number" dataKey="x" name="Dimension 1" hide />
          <YAxis type="number" dataKey="y" name="Dimension 2" hide />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="bg-slate-900 border border-slate-700 p-2 rounded shadow-xl text-xs">
                    <p className="text-slate-300">Frame: {payload[0].payload.frame}</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Scatter name="Trajectory" data={data} line={{ stroke: color, strokeWidth: 2 }} lineType="fitting">
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={color} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      
      {/* Legend overlay */}
      <div className="absolute bottom-4 right-4 text-[10px] text-slate-500 flex flex-col items-end gap-1">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
          <span>Straight = Natural</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-rose-500"></span>
          <span>Curved/Erratic = AI</span>
        </div>
      </div>
    </div>
  );
};

export default TrajectoryChart;
