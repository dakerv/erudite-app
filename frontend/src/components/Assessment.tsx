import { motion } from 'framer-motion';
import type { Assessment as AssessmentReport } from '../types/detection';
import { VERDICT_LABELS } from '../types/detection';

interface AssessmentProps {
  report: AssessmentReport;
  isSample: boolean;
  onCheckAnother: () => void;
}

export function Assessment({
  report,
  isSample,
  onCheckAnother
}: AssessmentProps) {
  return (
    <section
      id="assessment"
      aria-labelledby="assessment-heading"
      className="border-b border-rule bg-paper">
      
      <div className="mx-auto max-w-spread px-5 py-12 sm:px-8 lg:px-12 lg:py-16">
        <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-ink pb-2">
          <div className="flex items-baseline gap-4">
            <button
              type="button"
              onClick={onCheckAnother}
              className="font-mono text-[10px] uppercase tracking-label text-gray-soft underline decoration-rule underline-offset-4 transition-colors duration-150 ease-out hover:text-ink">
              
              ← Assessment / {report.id}
            </button>
            <p className="font-mono text-[10px] uppercase tracking-label text-ink">
              Image assessment / {report.id}
            </p>
          </div>
          <p className="font-mono text-[10px] uppercase tracking-label text-gray-soft">
            {isSample ?
            'Specimen report · Illustrative example' :
            `Source file · ${report.fileName}`}
          </p>
        </div>

        <div className="mt-6 grid gap-8 lg:grid-cols-12 lg:gap-14">
          <figure className="lg:col-span-5">
            <div className="paper-grain relative bg-ivorydeep">
              <img
                key={report.imageUrl}
                src={report.imageUrl}
                alt="The submitted image under assessment."
                className="h-[320px] w-full object-cover sm:h-[440px] lg:h-[520px]" />
              
            </div>
            <figcaption className="mt-3 border-t border-rule pt-2 font-mono text-[10px] uppercase tracking-label text-gray-soft">
              Submitted image · Examined at 224 × 224 px
            </figcaption>
          </figure>

          <div className="lg:col-span-7 lg:pl-2">
            <h2
              id="assessment-heading"
              className="font-display uppercase leading-[0.95] tracking-[-0.015em] text-ink"
              style={{ fontSize: 'clamp(2.2rem, 4.4vw, 4rem)' }}>
              
              {VERDICT_LABELS[report.predicted]}
            </h2>

            <p className="mt-4 border-b border-rule pb-6 font-mono text-[12px] uppercase tracking-label text-graphite">
              {report.confidence.toFixed(1)}% model confidence
            </p>

            <p className="mt-6 font-mono text-[10px] uppercase tracking-label text-gray-soft">
              Confidence distribution
            </p>
            <ul className="mt-4 space-y-5">
              {report.scores.map((score) => {
                const isTop = score.id === report.predicted;
                return (
                  <li key={score.id}>
                    <div className="flex items-baseline justify-between">
                      <span
                        className={`font-mono text-[11px] uppercase tracking-label ${
                        isTop ? 'text-ink' : 'text-gray-soft'}`
                        }>
                        
                        {score.label}
                      </span>
                      <span
                        className={`font-mono text-[11px] tracking-label ${
                        isTop ? 'text-ink' : 'text-gray-soft'}`
                        }>
                        
                        {score.probability.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-2 h-[3px] w-full bg-ivorydeep">
                      <motion.div
                        key={`${report.id}-${score.id}`}
                        className={`h-full ${isTop ? 'bg-ink' : 'bg-slate-tint'}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${score.probability}%` }}
                        transition={{
                          duration: 0.28,
                          ease: [0.23, 1, 0.32, 1]
                        }} />
                      
                    </div>
                  </li>);

              })}
            </ul>

            <div className="mt-8 grid gap-6 border-t border-rule pt-5 sm:grid-cols-12 sm:items-start">
              <p className="max-w-[52ch] text-[14px] leading-[1.65] text-graphite sm:col-span-7">
                Automated prediction. Results should be interpreted as an
                indication rather than definitive proof of authenticity.
              </p>
              <div className="sm:col-span-5 sm:text-right">
                <button
                  type="button"
                  onClick={onCheckAnother}
                  className="bg-ink px-6 py-3 font-mono text-[10px] uppercase tracking-label text-paper transition-colors duration-150 ease-out hover:bg-graphite">
                  
                  Check another image
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>);

}