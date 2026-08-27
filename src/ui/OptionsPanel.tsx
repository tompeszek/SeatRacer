import type { ControlState } from './options'
import {
  CLOSE_RACES_OPTIONS,
  LOSS_OPTIONS,
  LP_P_OPTIONS,
  RECENCY_OPTIONS,
  SHRINKAGE_OPTIONS,
  STERN_BIAS_OPTIONS,
  STRENGTH_OPTIONS,
} from './options'

interface PillRowProps {
  label: string
  options: string[]
  active: string
  caption?: string
  onSelect: (key: string) => void
}

function PillRow({ label, options, active, caption, onSelect }: PillRowProps) {
  return (
    <div className="opt-row">
      <span className="opt-label">{label}</span>
      <div className="pills">
        {options.map((key) => (
          <button
            key={key}
            className={`pill${key === active ? ' active' : ''}`}
            onClick={() => onSelect(key)}
          >
            {key}
          </button>
        ))}
      </div>
      {caption && <span className="opt-caption">{caption}</span>}
    </div>
  )
}

interface Props {
  controls: ControlState
  allShells: string[]
  onChange: (next: ControlState) => void
}

export function OptionsSection(props: {
  controls: ControlState
  allShells: string[]
  onControls: (c: ControlState) => void
}) {
  return <OptionsPanel controls={props.controls} allShells={props.allShells} onChange={props.onControls} />
}

export function OptionsPanel({ controls, allShells, onChange }: Props) {
  const set = (patch: Partial<ControlState>) => onChange({ ...controls, ...patch })
  const selectedShells = controls.shells ?? allShells

  return (
    <div className="options-box">
      <PillRow
        label="Error Scoring"
        options={Object.keys(LOSS_OPTIONS)}
        active={controls.loss}
        caption={LOSS_OPTIONS[controls.loss].caption}
        onSelect={(loss) => set({ loss: loss as ControlState['loss'] })}
      />
      {controls.loss === 'Lp' && (
        <PillRow
          label="Exponent"
          options={[...LP_P_OPTIONS]}
          active={controls.lpP}
          caption="p = 2 is squared error; p = 1 is absolute error"
          onSelect={(lpP) => set({ lpP: lpP as ControlState['lpP'] })}
        />
      )}
      <PillRow
        label="Shrinkage"
        options={Object.keys(SHRINKAGE_OPTIONS)}
        active={controls.shrinkage}
        caption={SHRINKAGE_OPTIONS[controls.shrinkage].caption}
        onSelect={(shrinkage) => set({ shrinkage: shrinkage as ControlState['shrinkage'] })}
      />
      {controls.shrinkage !== 'Off' && (
        <PillRow
          label="Strength"
          options={Object.keys(STRENGTH_OPTIONS)}
          active={controls.strength}
          caption={STRENGTH_OPTIONS[controls.strength].caption}
          onSelect={(strength) => set({ strength: strength as ControlState['strength'] })}
        />
      )}
      <PillRow
        label="Recency"
        options={Object.keys(RECENCY_OPTIONS)}
        active={controls.recency}
        caption={RECENCY_OPTIONS[controls.recency].caption}
        onSelect={(recency) => set({ recency: recency as ControlState['recency'] })}
      />
      <PillRow
        label="Close Races"
        options={Object.keys(CLOSE_RACES_OPTIONS)}
        active={controls.close}
        caption={CLOSE_RACES_OPTIONS[controls.close].caption}
        onSelect={(close) => set({ close: close as ControlState['close'] })}
      />
      <PillRow
        label="Stern Bias"
        options={Object.keys(STERN_BIAS_OPTIONS)}
        active={controls.stern}
        caption={STERN_BIAS_OPTIONS[controls.stern].caption}
        onSelect={(stern) => set({ stern: stern as ControlState['stern'] })}
      />
      <PillRow
        label="Coxswains"
        options={['Include', 'Exclude']}
        active={controls.coxswains ? 'Include' : 'Exclude'}
        caption="Whether coxswains get their own coefficient"
        onSelect={(key) => set({ coxswains: key === 'Include' })}
      />
      <div className="opt-row">
        <span className="opt-label">Shells</span>
        <div className="pills">
          {allShells.map((shell) => {
            const active = selectedShells.includes(shell)
            return (
              <button
                key={shell}
                className={`pill${active ? ' active' : ''}`}
                onClick={() => {
                  const next = active
                    ? selectedShells.filter((s) => s !== shell)
                    : [...selectedShells, shell]
                  set({ shells: next.length === allShells.length ? null : next })
                }}
              >
                {shell}
              </button>
            )
          })}
        </div>
        <span className="opt-caption">Shell classes included in the fit</span>
      </div>
    </div>
  )
}
