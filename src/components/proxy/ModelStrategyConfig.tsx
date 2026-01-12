import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { 
    Plus, 
    Trash2, 
    Edit2, 
    Save, 
    Settings,
    ArrowUp,
    ArrowDown
} from 'lucide-react';
import { 
    ModelStrategy, 
    ModelPriority, 
    ModelStickiness 
} from '../../types/config';
import ModalDialog from '../common/ModalDialog';
import { useProxyModels } from '../../hooks/useProxyModels';

interface ModelStrategyConfigProps {
    strategies: Record<string, ModelStrategy>;
    onUpdate: (strategies: Record<string, ModelStrategy>) => void;
}

export default function ModelStrategyConfig({ strategies = {}, onUpdate }: ModelStrategyConfigProps) {
    const { t } = useTranslation();
    const { models } = useProxyModels();
    
    // State
    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [tempStrategy, setTempStrategy] = useState<ModelStrategy>({
        candidates: [],
        policy: {
            model_priority: ModelPriority.AccuracyFirst,
            stickiness: ModelStickiness.Strong,
            max_model_hops: 1
        }
    });
    const [tempId, setTempId] = useState('');
    const [newCandidate, setNewCandidate] = useState('');

    const handleEdit = (id: string) => {
        setEditingId(id);
        setTempId(id);
        setTempStrategy(JSON.parse(JSON.stringify(strategies[id])));
        setIsEditModalOpen(true);
    };

    const handleCreate = () => {
        setEditingId(null);
        setTempId(''); // New ID input
        setTempStrategy({
            candidates: [],
            policy: {
                model_priority: ModelPriority.AccuracyFirst,
                stickiness: ModelStickiness.Strong,
                max_model_hops: 1
            }
        });
        setIsEditModalOpen(true);
    };

    const handleDelete = (id: string) => {
        // eslint-disable-next-line no-restricted-globals
        if (confirm(t('common.confirm_delete', { defaultValue: 'Are you sure you want to delete this strategy?' }))) {
            const newStrategies = { ...strategies };
            delete newStrategies[id];
            onUpdate(newStrategies);
        }
    };

    const handleSave = () => {
        if (!tempId.trim()) return;
        
        const newStrategies = { ...strategies };
        
        // If renaming, delete old key
        if (editingId && editingId !== tempId) {
            delete newStrategies[editingId];
        }
        
        newStrategies[tempId] = tempStrategy;
        onUpdate(newStrategies);
        setIsEditModalOpen(false);
    };

    const addCandidate = () => {
        if (newCandidate && !tempStrategy.candidates.includes(newCandidate)) {
            setTempStrategy({
                ...tempStrategy,
                candidates: [...tempStrategy.candidates, newCandidate]
            });
            setNewCandidate('');
        }
    };

    const removeCandidate = (index: number) => {
        const newCandidates = [...tempStrategy.candidates];
        newCandidates.splice(index, 1);
        setTempStrategy({
            ...tempStrategy,
            candidates: newCandidates
        });
    };

    const moveCandidate = (index: number, direction: 'up' | 'down') => {
        const newCandidates = [...tempStrategy.candidates];
        if (direction === 'up' && index > 0) {
            [newCandidates[index - 1], newCandidates[index]] = [newCandidates[index], newCandidates[index - 1]];
        } else if (direction === 'down' && index < newCandidates.length - 1) {
            [newCandidates[index], newCandidates[index + 1]] = [newCandidates[index + 1], newCandidates[index]];
        }
        setTempStrategy({
            ...tempStrategy,
            candidates: newCandidates
        });
    };

    return (
        <div className="space-y-4">
            <div className="flex justify-between items-center mb-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                    {t('proxy.strategies.description', { defaultValue: 'Define advanced routing strategies with candidate pools and fallback policies.' })}
                </div>
                <button
                    onClick={handleCreate}
                    className="btn btn-sm btn-primary gap-2"
                >
                    <Plus size={14} />
                    {t('common.add', { defaultValue: 'Add Strategy' })}
                </button>
            </div>

            {Object.keys(strategies).length === 0 ? (
                <div className="text-center py-8 text-gray-400 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg">
                    {t('proxy.strategies.empty', { defaultValue: 'No strategies defined' })}
                </div>
            ) : (
                <div className="grid grid-cols-1 gap-3">
                    {Object.entries(strategies).map(([id, strategy]) => (
                        <div 
                            key={id} 
                            className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-100 dark:border-gray-700 hover:border-blue-200 dark:hover:border-blue-800 transition-colors"
                        >
                            <div className="flex-1 min-w-0 mr-4">
                                <div className="flex items-center gap-2 mb-1">
                                    <h4 className="font-mono font-medium text-sm text-blue-600 dark:text-blue-400 truncate">
                                        {id}
                                    </h4>
                                    <span className="badge badge-xs badge-ghost font-normal">
                                        {strategy.candidates.length} candidates
                                    </span>
                                </div>
                                <div className="text-xs text-gray-500 dark:text-gray-400 flex gap-3">
                                    <span className="flex items-center gap-1">
                                        <ArrowUp size={10} />
                                        {strategy.policy.model_priority === ModelPriority.AccuracyFirst ? 'Accuracy' : 'Capacity'}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <Settings size={10} />
                                        {strategy.policy.stickiness === ModelStickiness.Strong ? 'Strong Stickiness' : 'Weak Stickiness'}
                                    </span>
                                </div>
                            </div>
                            
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => handleEdit(id)}
                                    className="p-1.5 text-gray-400 hover:text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-md transition-colors"
                                    title={t('common.edit', { defaultValue: 'Edit' })}
                                >
                                    <Edit2 size={14} />
                                </button>
                                <button
                                    onClick={() => handleDelete(id)}
                                    className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-md transition-colors"
                                    title={t('common.delete', { defaultValue: 'Delete' })}
                                >
                                    <Trash2 size={14} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Edit Modal */}
            <ModalDialog
                isOpen={isEditModalOpen}
                onCancel={() => setIsEditModalOpen(false)}
                title={editingId ? t('proxy.strategies.edit_title', { defaultValue: 'Edit Strategy' }) : t('proxy.strategies.create_title', { defaultValue: 'New Strategy' })}
                message=""
                onConfirm={handleSave}
                footer={
                    <div className="flex justify-end gap-3">
                        <button
                            className="btn btn-ghost btn-sm"
                            onClick={() => setIsEditModalOpen(false)}
                        >
                            {t('common.cancel', { defaultValue: 'Cancel' })}
                        </button>
                        <button
                            className="btn btn-primary btn-sm gap-2"
                            onClick={handleSave}
                            disabled={!tempId.trim() || tempStrategy.candidates.length === 0}
                        >
                            <Save size={14} />
                            {t('common.save', { defaultValue: 'Save' })}
                        </button>
                    </div>
                }
            >
                <div className="space-y-4">
                    {/* Strategy ID */}
                    <div className="form-control w-full">
                        <label className="label">
                            <span className="label-text font-medium">{t('proxy.strategies.id_label', { defaultValue: 'Strategy ID' })}</span>
                        </label>
                        <input
                            type="text"
                            placeholder="e.g., fast-fallback-strategy"
                            className="input input-bordered w-full font-mono text-sm"
                            value={tempId}
                            onChange={(e) => setTempId(e.target.value)}
                            // disabled={!!editingId} // Allow renaming?
                        />
                        <label className="label">
                            <span className="label-text-alt text-gray-500">
                                {t('proxy.strategies.id_hint', { defaultValue: 'Unique identifier for referring to this strategy.' })}
                            </span>
                        </label>
                    </div>

                    <div className="divider my-0"></div>

                    {/* Policy Configuration */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="form-control w-full">
                            <label className="label">
                                <span className="label-text">{t('proxy.strategies.priority_label', { defaultValue: 'Priority' })}</span>
                            </label>
                            <select 
                                className="select select-bordered w-full"
                                value={tempStrategy.policy.model_priority}
                                onChange={(e) => setTempStrategy({
                                    ...tempStrategy,
                                    policy: { ...tempStrategy.policy, model_priority: e.target.value as ModelPriority }
                                })}
                            >
                                <option value={ModelPriority.AccuracyFirst}>Accuracy First</option>
                                <option value={ModelPriority.CapacityFirst}>Capacity First</option>
                            </select>
                        </div>

                        <div className="form-control w-full">
                            <label className="label">
                                <span className="label-text">{t('proxy.strategies.stickiness_label', { defaultValue: 'Stickiness' })}</span>
                            </label>
                            <select 
                                className="select select-bordered w-full"
                                value={tempStrategy.policy.stickiness}
                                onChange={(e) => setTempStrategy({
                                    ...tempStrategy,
                                    policy: { ...tempStrategy.policy, stickiness: e.target.value as ModelStickiness }
                                })}
                            >
                                <option value={ModelStickiness.Strong}>Strong</option>
                                <option value={ModelStickiness.Weak}>Weak</option>
                            </select>
                        </div>
                    </div>

                    {/* Candidates */}
                    <div className="form-control w-full">
                        <label className="label">
                            <span className="label-text font-medium">{t('proxy.strategies.candidates_label', { defaultValue: 'Candidate Models (Ordered)' })}</span>
                        </label>
                        
                        <div className="flex gap-2 mb-2">
                             <input
                                list="available-models"
                                type="text"
                                className="input input-bordered w-full flex-1"
                                placeholder="Type or select model ID..."
                                value={newCandidate}
                                onChange={(e) => setNewCandidate(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter') {
                                        e.preventDefault();
                                        addCandidate();
                                    }
                                }}
                            />
                            <datalist id="available-models">
                                {models.map(m => (
                                    <option key={m.id} value={m.id}>{m.name}</option>
                                ))}
                            </datalist>
                            <button 
                                className="btn btn-square btn-outline"
                                onClick={addCandidate}
                                disabled={!newCandidate}
                            >
                                <Plus size={18} />
                            </button>
                        </div>

                        <div className="bg-base-200/50 dark:bg-base-300/30 rounded-lg p-2 min-h-[100px] max-h-[200px] overflow-y-auto space-y-1 border border-base-200 dark:border-base-300">
                             {tempStrategy.candidates.length === 0 ? (
                                <div className="text-gray-400 text-sm text-center py-8">
                                    Add at least one candidate model
                                </div>
                             ) : (
                                tempStrategy.candidates.map((cand, idx) => (
                                    <div key={idx} className="flex items-center justify-between bg-white dark:bg-base-100 px-3 py-2 rounded border border-gray-100 dark:border-gray-700 shadow-sm">
                                        <div className="flex items-center gap-2">
                                            <span className="badge badge-sm badge-neutral w-5 h-5 flex items-center justify-center p-0">{idx + 1}</span>
                                            <span className="font-mono text-sm">{cand}</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <button 
                                                className="btn btn-ghost btn-xs btn-square"
                                                disabled={idx === 0}
                                                onClick={() => moveCandidate(idx, 'up')}
                                            >
                                                <ArrowUp size={14} />
                                            </button>
                                            <button 
                                                className="btn btn-ghost btn-xs btn-square"
                                                disabled={idx === tempStrategy.candidates.length - 1}
                                                onClick={() => moveCandidate(idx, 'down')}
                                            >
                                                <ArrowDown size={14} />
                                            </button>
                                            <button 
                                                className="btn btn-ghost btn-xs btn-square text-error"
                                                onClick={() => removeCandidate(idx)}
                                            >
                                                <Trash2 size={14} />
                                            </button>
                                        </div>
                                    </div>
                                ))
                             )}
                        </div>
                    </div>

                </div>
            </ModalDialog>
        </div>
    );
}
