import { ModelComparisonBar } from "@/components/ui/ModelComparisonBar";

export default function ExperimentResults() {
  const evaluationResults = {
    "random_forest_v2": 0.89,
    "xgboost_tuned": 0.94,
    "logistic_regression": 0.76,
  };

  return (
    <div className="max-w-md p-6 bg-surface-1 rounded-xl shadow-sm">
      <ModelComparisonBar 
        allScores={evaluationResults}
        winnerName="xgboost_tuned"
        metricName="F1-Score"
      />
    </div>
  );
}