'use client';

import { useState, useMemo, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Users, Filter, Database, TrendingUp, Play, RotateCcw, Download, ClipboardList, GitBranch, Sun, Moon, Upload } from 'lucide-react';
import { io, Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import { API_CONFIG } from '@/lib/api-config';
import ExperienceGenerator from './ExperienceGenerator';
import MultiAgentSimulator from './MultiAgentSimulator';
import PostProcessor from './PostProcessor';
import DatasetConstructor from './DatasetConstructor';
import Evaluator from './Evaluator';
import Versions from './Versions';
import HumanEvaluation from './HumanEvaluation';

/** Matches backend `validate_dialogue_format` + comprehensive evaluation pipeline. */
const EVALUATION_DIALOGUES_JSON_SCHEMA = `{
  "dialogues": [
    {
      "dialogue_id": "unique-id-string",
      "goal": "Task the user wants completed",
      "domain": "hotel",
      "context": "optional — scenario details",
      "turns": [
        { "role": "User", "text": "…", "timestamp": "optional ISO8601" },
        { "role": "SupportBot", "text": "…", "timestamp": "optional" }
      ],
      "metadata": {},
      "goal_data": {}
    }
  ]
}

You may also upload a raw JSON array: [ { "dialogue_id": "…", "goal": "…", "domain": "…", "turns": [ … ] }, … ]

Required per dialogue: dialogue_id, goal, domain, turns.
Each turn: role must be "User" or "SupportBot", and "text" is required.
Optional: timestamp on turns (used for response-time metrics). Up to 500 dialogues per request.`;

interface Experience {
  experience_id: string;
  domain: string;
  task: string;
  personas: Array<{
    role: string;
    name: string;
    traits: string[];
    memory?: string[];
  }>;
  situation: string;
  goal: string;
  constraints: {
    max_turns: number;
    max_tokens_per_turn: number;
  };
  conversation_starter: string;
}

interface Conversation {
  conv_id: string;
  domain: string;
  task: string;
  experience_id: string;
  personas: Array<{
    role: string;
    name: string;
    traits: string[];
    memory?: string[];
  }>;
  situation: string;
  goal: string;
  constraints: {
    max_turns: number;
    max_tokens_per_turn: number;
  };
  conversation_starter: string;
  turns: Array<{
    turn_id: number;
    speaker: string;
    text: string;
  }>;
  task_success: boolean;
  judge_score: number;
  mtld: number;
  provenance: {
    generator_model: string;
    prompt_version: string;
    temperature: number;
    shot_ids: string[];
    timestamp: string;
  };
}

interface FilteredConversation {
  id: string;
  original_id: string;
  status: 'kept' | 'removed' | 'modified';
  reason: string;
  score: number;
  metadata: {
    similarity_score?: number;
    fluency_score?: number;
    coherence_score?: number;
    task_success_score?: number;
  };
}

interface DatasetItem {
  id: string;
  conv_id: string;
  domain: string;
  task: string;
  personas: Array<{
    name: string;
    role: string;
    traits: string[];
  }>;
  turns: Array<{
    speaker: string;
    text: string;
    turn_id: number;
  }>;
  task_success: boolean;
  metadata: {
    total_turns: number;
    domain_category: string;
    creation_timestamp: string;
    quality_score: number;
  };
}

interface EvaluationMetrics {
  overall_score: number;
  diversity_score: number;
  coherence_score: number;
  task_success_rate: number;
  fluency_score: number;
  groundedness_score: number;
  categories: {
    lexical_diversity: number;
    conversation_length: {
      avg_turns: number;
      std_dev: number;
    };
    domain_distribution: Record<string, number>;
    task_success_by_domain: Record<string, number>;
    goal_completion_by_domain?: Record<string, number>;
  };
}

interface PipelineStep {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  status: 'pending' | 'active' | 'completed' | 'error';
  progress: number;
  component: React.ReactNode;
}

interface RequestLogEntry {
  id: string;
  stepId: string;
  stepName: string;
  endpoint: string;
  status: 'success' | 'error';
  message: string;
  timestamp: string;
}

export default function GoalConvoDashboard() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [pipelineRunId, setPipelineRunId] = useState(0);
  const [pipelineData, setPipelineData] = useState<{
    experiences: Experience[];
    conversations: Conversation[];
    filteredConversations: FilteredConversation[];
    dataset: DatasetItem[];
    evaluations: EvaluationMetrics | null;
  }>({
    experiences: [],
    conversations: [],
    filteredConversations: [],
    dataset: [],
    evaluations: null
  });
  const [requestLogs, setRequestLogs] = useState<RequestLogEntry[]>([]);
  const socketRef = useRef<Socket | null>(null);
  const sessionIdRef = useRef<string>(uuidv4());
  
  // Pipeline configuration state
  const [selectedDomains, setSelectedDomains] = useState<string[]>(['hotel']);
  const [numDialogues, setNumDialogues] = useState<number>(1);
  // Ablation / experiment options (optional)
  const [experimentTag, setExperimentTag] = useState<string>('');
  const [ablationQualityJudgeOff, setAblationQualityJudgeOff] = useState<boolean>(false);
  const [ablationFewShot, setAblationFewShot] = useState<string>('');
  const [ablationTemperature, setAblationTemperature] = useState<string>('');

  // Top-level tabs: Pipeline | Versions | Human Evaluation
  const [activeTab, setActiveTab] = useState<'pipeline' | 'versions' | 'human-eval'>('pipeline');

  // Light / Dark mode (persisted in localStorage)
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window === 'undefined') return 'dark';
    return (localStorage.getItem('goalconvo-theme') as 'light' | 'dark') || 'dark';
  });
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('goalconvo-theme', theme);
    } catch (_) {}
  }, [theme]);

  // Live dialogue viewer (during pipeline run)
  const [liveDialogue, setLiveDialogue] = useState<{
    current_turns: Array<{ role: string; text: string }>;
    step_message: string;
    dialogue_index?: number;
    total_dialogues?: number;
    goal?: string;
  } | null>(null);
  
  // Evaluation step: live sub-step messages (e.g. "Computing BERTScore...")
  const [evaluationStepMessages, setEvaluationStepMessages] = useState<string[]>([]);
  // Separate evaluation run (after dialogue generation is done)
  const [isEvaluationRunning, setIsEvaluationRunning] = useState(false);

  // Backend connection status
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [backendCheckInProgress, setBackendCheckInProgress] = useState<boolean>(false);
  
  // Available domains
  const availableDomains = ['hotel', 'restaurant', 'taxi', 'train', 'attraction'];

  // Evaluation configuration (separate from generation)
  const [evaluationDomains, setEvaluationDomains] = useState<string[]>([]);
  const [evaluationLimit, setEvaluationLimit] = useState<number>(10);
  const [uploadedEvaluationDialogues, setUploadedEvaluationDialogues] = useState<unknown[]>([]);
  const [uploadedEvaluationFileNames, setUploadedEvaluationFileNames] = useState<string[]>([]);

  // Check backend health on mount and periodically
  useEffect(() => {
    const checkBackendHealth = async () => {
      setBackendCheckInProgress(true);
      try {
        const isHealthy = await API_CONFIG.checkHealth();
        setBackendConnected(isHealthy);
        if (isHealthy) {
          console.log('✅ Backend is healthy');
        } else {
          console.warn('⚠️ Backend health check failed');
        }
      } catch (error) {
        console.error('❌ Backend health check error:', error);
        setBackendConnected(false);
      } finally {
        setBackendCheckInProgress(false);
      }
    };

    // Check immediately
    checkBackendHealth();

    // Check every 10 seconds
    const healthCheckInterval = setInterval(checkBackendHealth, 10000);

    return () => clearInterval(healthCheckInterval);
  }, []);

  // Setup WebSocket connection
  useEffect(() => {
    const socketUrl = API_CONFIG.getSocketUrl();
    console.log('🔌 Connecting to WebSocket:', socketUrl);
    
    const socket = io(socketUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      timeout: 20000,
    });
    
    socketRef.current = socket;
    
    // Connection status tracking
    socket.on('connect', () => {
      console.log('✅ WebSocket connected:', socket.id);
      setBackendConnected(true);
      addRequestLog({
        stepId: 'connection',
        stepName: 'WebSocket Connection',
        endpoint: '/socket',
        status: 'success',
        message: `Connected to pipeline server (ID: ${socket.id})`
      });
      
      // Join session immediately after connection
      socket.emit('join_session', { session_id: sessionIdRef.current });
    });
    
    socket.on('disconnect', (reason: string) => {
      console.log('❌ WebSocket disconnected:', reason);
      setBackendConnected(false);
      addRequestLog({
        stepId: 'connection',
        stepName: 'WebSocket Connection',
        endpoint: '/socket',
        status: 'error',
        message: `Disconnected: ${reason}`
      });
    });
    
    socket.on('connect_error', (error: Error) => {
      console.error('❌ WebSocket connection error:', error);
      setBackendConnected(false);
      const currentSocketUrl = API_CONFIG.getSocketUrl();
      addRequestLog({
        stepId: 'connection',
        stepName: 'WebSocket Connection',
        endpoint: '/socket',
        status: 'error',
        message: `Connection error: ${error.message}. Make sure the backend server is running on ${currentSocketUrl}`
      });
    });
    
    socket.on('reconnect', (attemptNumber: number) => {
      console.log('🔄 WebSocket reconnected after', attemptNumber, 'attempts');
      addRequestLog({
        stepId: 'connection',
        stepName: 'WebSocket Connection',
        endpoint: '/socket',
        status: 'success',
        message: `Reconnected after ${attemptNumber} attempts`
      });
    });
    
    // Server confirmation events
    socket.on('connected', (data: any) => {
      console.log('✅ Server confirmed connection:', data);
      addRequestLog({
        stepId: 'connection',
        stepName: 'WebSocket Connection',
        endpoint: '/socket',
        status: 'success',
        message: `Server confirmed: ${data.message || 'Connected'}`
      });
    });
    
    socket.on('joined', (data: any) => {
      console.log('✅ Joined session:', data);
      addRequestLog({
        stepId: 'connection',
        stepName: 'Session Join',
        endpoint: '/socket',
        status: 'success',
        message: `Joined session: ${data.session_id || 'unknown'}`
      });
    });
    
    // Pipeline events
    socket.on('pipeline_start', (data: any) => {
      console.log('Pipeline started:', data);
      addRequestLog({
        stepId: 'pipeline',
        stepName: 'Pipeline Start',
        endpoint: '/api/run-pipeline',
        status: 'success',
        message: data.data?.message || 'Pipeline started'
      });
      setIsRunning(true);
      setCurrentStep(0);
      setEvaluationStepMessages([]);
    });
    
    socket.on('step_start', (data: any) => {
      console.log('Step started:', data);
      const stepData = data.data || {};
      const stepName = stepData.step_name || stepData.step || 'Unknown Step';
      
      addRequestLog({
        stepId: stepData.step || 'unknown',
        stepName: stepName,
        endpoint: '/api/run-pipeline',
        status: 'success',
        message: stepData.message || `${stepName} started`
      });
      
      // Update current step: backend has 5 steps (incl. saving); UI has 4 (Dataset Construction commented out).
      // Map backend step to frontend index 0–3 so Evaluation (UI index 3) gets the circling border.
      const stepMap: Record<string, number> = {
        'experience_generation': 0,
        'dialogue_simulation': 1,
        'quality_filtering': 2,
        'saving': 3,
        'evaluation': 3
      };
      
      if (stepData.step && stepMap[stepData.step] !== undefined) {
        setCurrentStep(stepMap[stepData.step]);
      }
    });
    
    socket.on('step_data', (data: any) => {
      console.log('Step data received:', data);
      const stepData = data.data || {};
      const step = data.data?.step || stepData.step || 'unknown';
      
      // Only process events for the current session
      const eventSessionId = data.session_id;
      if (eventSessionId && eventSessionId !== sessionIdRef.current) {
        console.log(`Ignoring step_data event for different session: ${eventSessionId} (current: ${sessionIdRef.current})`);
        return;
      }
      
      // Update pipeline data based on step
      if (step === 'experience_generation' && stepData.data?.experience) {
        const exp = stepData.data.experience;
        const userPersona = exp.user_persona;
        const userPersonaName = typeof userPersona === 'string'
          ? userPersona
          : (userPersona && typeof userPersona === 'object' && 'name' in userPersona
            ? String((userPersona as { name?: string }).name || 'User')
            : 'User');
        const experience: Experience = {
          experience_id: exp.experience_id || `exp_${Date.now()}`,
          domain: exp.domain || 'unknown',
          task: exp.goal || exp.task || '',
          personas: [
            { role: 'user', name: userPersonaName, traits: [], memory: [] },
            { role: 'assistant', name: 'SupportBot', traits: ['helpful', 'professional'], memory: [] }
          ],
          situation: exp.context || '',
          goal: exp.goal || '',
          constraints: { max_turns: 10, max_tokens_per_turn: 100 },
          conversation_starter: exp.first_utterance || ''
        };
        
        setPipelineData(prev => {
          // Check if this experience already exists (prevent duplicates)
          const existingIndex = prev.experiences.findIndex(
            e => e.experience_id === experience.experience_id
          );
          
          if (existingIndex >= 0) {
            console.log(`Experience ${experience.experience_id} already exists, skipping duplicate`);
            return prev; // Don't add duplicate
          }
          
          return {
            ...prev,
            experiences: [...prev.experiences, experience]
          };
        });
      }
      
      if (step === 'dialogue_simulation' && stepData.data?.dialogue) {
        const dial = stepData.data.dialogue;
        console.log('Processing dialogue:', dial);
        console.log('Dialogue turns:', dial.turns);
        
        // Convert backend dialogue format to frontend conversation format
        // Ensure turns is an array
        let turnsArray: any[] = [];
        if (Array.isArray(dial.turns)) {
          turnsArray = dial.turns;
        } else if (dial.turns && typeof dial.turns === 'object') {
          // If turns is an object, try to convert it
          turnsArray = Object.values(dial.turns);
        }
        
        console.log('Processed turns array:', turnsArray);
        
        const dialUserPersona = dial.user_persona;
        const dialUserPersonaName = typeof dialUserPersona === 'string'
          ? dialUserPersona
          : (dialUserPersona && typeof dialUserPersona === 'object' && 'name' in dialUserPersona
            ? String((dialUserPersona as { name?: string }).name || 'User')
            : 'User');
        const conversation: Conversation = {
          conv_id: dial.dialogue_id || `conv_${Date.now()}`,
          domain: dial.domain || 'unknown',
          task: dial.goal || '',
          experience_id: dial.experience_id || '',
          personas: [
            { role: 'user', name: dialUserPersonaName, traits: [], memory: [] },
            { role: 'assistant', name: 'SupportBot', traits: ['helpful', 'professional'], memory: [] }
          ],
          situation: dial.context || '',
          goal: dial.goal || '',
          constraints: { max_turns: 10, max_tokens_per_turn: 100 },
          conversation_starter: turnsArray[0]?.text || '',
          turns: turnsArray.map((turn: any, idx: number) => ({
            turn_id: idx + 1,
            speaker: turn.role === 'User' ? 'user' : 'assistant',
            text: turn.text || ''
          })),
          task_success: dial.metadata?.quality_score > 0.7 || false,
          judge_score: (dial.metadata?.quality_score || 0) * 5,
          mtld: 0,
          provenance: {
            generator_model: dial.metadata?.model_version || 'unknown',
            prompt_version: 'v1',
            temperature: 0.7,
            shot_ids: [],
            timestamp: dial.metadata?.generated_at || new Date().toISOString()
          }
        };
        
        console.log('Adding conversation to pipeline data:', conversation);
        setPipelineData(prev => {
          // Check if this conversation already exists (prevent duplicates)
          const existingIndex = prev.conversations.findIndex(
            c => c.conv_id === conversation.conv_id
          );
          
          if (existingIndex >= 0) {
            console.log(`Conversation ${conversation.conv_id} already exists, skipping duplicate`);
            return prev; // Don't add duplicate
          }
          
          const updated = {
            ...prev,
            conversations: [...prev.conversations, conversation]
          };
          console.log('Updated pipeline data conversations count:', updated.conversations.length);
          return updated;
        });
      }
      
      if (step === 'quality_filtering' && stepData.data) {
        const filterData = stepData.data;
        addRequestLog({
          stepId: 'quality_filtering',
          stepName: 'Quality Filtering',
          endpoint: '/api/run-pipeline',
          status: 'success',
          message: `Accepted: ${filterData.accepted || 0}, Rejected: ${filterData.rejected || 0}`
        });
      }
    });
    
    socket.on('live_dialogue', (data: any) => {
      const payload = data.data || {};
      if (data.session_id && data.session_id !== sessionIdRef.current) return;
      setLiveDialogue({
        current_turns: Array.isArray(payload.current_turns) ? payload.current_turns : [],
        step_message: payload.step_message || 'Generating...',
        dialogue_index: payload.dialogue_index,
        total_dialogues: payload.total_dialogues,
        goal: payload.goal
      });
    });
    
    socket.on('log', (data: any) => {
      console.log('Log received:', data);
      const logData = data?.data ?? data ?? {};
      const step = logData.step || 'pipeline';
      const message = logData.message ?? '';
      const isError =
        logData.level === 'error' ||
        /error|failed|exception/i.test(String(message));
      const stepNames: Record<string, string> = {
        evaluation: 'Evaluation',
        experience_generation: 'Experience Generation',
        dialogue_simulation: 'Dialogue Simulation',
        quality_filtering: 'Quality Filtering',
        saving: 'Saving',
        pipeline: 'Pipeline',
      };
      addRequestLog({
        stepId: step,
        stepName: stepNames[step] || step,
        endpoint: step === 'evaluation' ? '/api/run-evaluation' : '/api/run-pipeline',
        status: isError ? 'error' : 'success',
        message: message || (isError ? 'Error' : 'OK')
      });
      if (step === 'evaluation' && message) {
        setEvaluationStepMessages((prev) => [...prev.slice(-14), message]);
      }
    });
    
    socket.on('pipeline_complete', (data: any) => {
      console.log('Pipeline completed:', data);
      
      // Only process events for the current session
      const eventSessionId = data.session_id;
      if (eventSessionId && eventSessionId !== sessionIdRef.current) {
        console.log(`Ignoring pipeline_complete event for different session: ${eventSessionId} (current: ${sessionIdRef.current})`);
        return;
      }
      
      const completeData = data.data || {};
      addRequestLog({
        stepId: 'pipeline',
        stepName: 'Pipeline Complete',
        endpoint: '/api/run-pipeline',
        status: 'success',
        message: completeData.message || 'Pipeline completed successfully'
      });
      
      // Store evaluation metrics if available
      if (completeData.evaluation) {
        console.log('Evaluation metrics received:', completeData.evaluation);
        setPipelineData(prev => ({
          ...prev,
          evaluations: completeData.evaluation
        }));
      }
      
      if (completeData.final_data) {
        // Update with final statistics
        console.log('Final data:', completeData.final_data);
      }
      
      setLiveDialogue(null);
      setIsRunning(false);
      setCurrentStep(2); // Dialogue generation done; Evaluation is separate (run via Run Evaluation button)
    });

    socket.on('evaluation_complete', (data: any) => {
      const eventSessionId = data.session_id;
      if (eventSessionId && eventSessionId !== sessionIdRef.current) return;
      const payload = data.data || {};
      if (payload.evaluation) {
        setPipelineData(prev => ({ ...prev, evaluations: payload.evaluation }));
      }
      addRequestLog({
        stepId: 'evaluation',
        stepName: 'Evaluation Complete',
        endpoint: '/api/run-evaluation',
        status: 'success',
        message: payload.message || 'Evaluation completed successfully'
      });
      setEvaluationStepMessages([]);
      setIsEvaluationRunning(false);
    });

    socket.on('evaluation_error', (data: any) => {
      const eventSessionId = data.session_id;
      if (eventSessionId && eventSessionId !== sessionIdRef.current) return;
      const payload = data.data || {};
      const msg = payload.message || payload.error || 'Evaluation failed';
      addRequestLog({
        stepId: 'evaluation',
        stepName: 'Evaluation Error',
        endpoint: '/api/run-evaluation',
        status: 'error',
        message: msg
      });
      setIsEvaluationRunning(false);
    });

    socket.on('pipeline_error', (data: any) => {
      console.error('Pipeline error:', data);
      const errorData = data?.data ?? data ?? {};
      const message =
        errorData.message ??
        errorData.error ??
        (typeof data?.message === 'string' ? data.message : null) ??
        (typeof data?.error === 'string' ? data.error : null) ??
        'Pipeline failed';
      addRequestLog({
        stepId: 'pipeline',
        stepName: 'Pipeline Error',
        endpoint: '/api/run-pipeline',
        status: 'error',
        message: String(message)
      });
      setLiveDialogue(null);
      setIsRunning(false);
    });
    
    
    // Cleanup on unmount
    return () => {
      console.log('Cleaning up WebSocket connection');
      socket.disconnect();
    };
  }, []);
  
  // Debug: Log when pipeline state changes
  useEffect(() => {
    console.log('Pipeline state changed:', { isRunning, currentStep, pipelineRunId });
  }, [isRunning, currentStep, pipelineRunId]);

  const addRequestLog = (entry: Omit<RequestLogEntry, 'id' | 'timestamp'>) => {
    const timestamp = new Date().toISOString();
    const id = `${timestamp}-${Math.random().toString(36).slice(2, 8)}`;
    setRequestLogs(prev => [
      { id, timestamp, ...entry },
      ...prev
    ].slice(0, 50));
  };

  const steps: PipelineStep[] = useMemo(() => [
    {
      id: 'experience',
      name: 'Experience Generation',
      description: 'Create structured blueprints with personas, situations, and goals',
      icon: <Brain className="w-6 h-6" />,
      status: 'pending',
      progress: 0,
      component: (
        <ExperienceGenerator
          key={`experience-${pipelineRunId}-${currentStep}`}
          experiences={pipelineData.experiences}
          autoStart={false}
          onComplete={(data) => handleStepComplete(0, data)}
          onLog={(log) =>
            addRequestLog({
              stepId: 'experience',
              stepName: 'Experience Generation',
              endpoint: '/api/run-pipeline',
              ...log,
            })
          }
        />
      )
    },
    {
      id: 'simulation',
      name: 'Multi-Agent Simulation',
      description: 'Generate natural dialogues through LLM agent interactions',
      icon: <Users className="w-6 h-6" />,
      status: 'pending',
      progress: 0,
      component: (
        <MultiAgentSimulator
          key={`simulation-${pipelineRunId}-${currentStep}`}
          experiences={pipelineData.experiences}
          conversations={pipelineData.conversations}
          autoStart={false}
          onComplete={(data) => handleStepComplete(1, data)}
          onLog={(log) =>
            addRequestLog({
              stepId: 'simulation',
              stepName: 'Multi-Agent Simulation',
              endpoint: '/api/run-pipeline',
              ...log,
            })
          }
        />
      )
    },
    {
      id: 'postprocessing',
      name: 'Post-Processing',
      description: 'Filter, deduplicate, and ensure quality of generated dialogues',
      icon: <Filter className="w-6 h-6" />,
      status: 'pending',
      progress: 0,
      component: (
        <PostProcessor
          key={`postprocessor-${pipelineRunId}-${currentStep}`}
          conversations={pipelineData.conversations}
          filteredConversations={pipelineData.filteredConversations}
          autoStart={false}
          onComplete={(data) => handleStepComplete(2, data)}
          onLog={(log) =>
            addRequestLog({
              stepId: 'postprocessing',
              stepName: 'Post-Processing',
              endpoint: '/api/run-pipeline',
              ...log,
            })
          }
        />
      )
    },
    // {
    //   id: 'dataset',
    //   name: 'Dataset Construction',
    //   description: 'Compile high-quality dialogues into structured dataset',
    //   icon: <Database className="w-6 h-6" />,
    //   status: 'pending',
    //   progress: 0,
    //   component: (
    //     <DatasetConstructor
    //       key={`dataset-${pipelineRunId}-${currentStep}`}
    //       filteredConversations={pipelineData.filteredConversations}
    //       dataset={pipelineData.dataset}
    //       conversations={pipelineData.conversations}
    //       autoStart={false}
    //       onComplete={(data) => handleStepComplete(3, data)}
    //       onLog={(log) =>
    //         addRequestLog({
    //           stepId: 'dataset',
    //           stepName: 'Dataset Construction',
    //           endpoint: '/api/run-pipeline',
    //           ...log,
    //         })
    //       }
    //     />
    //   )
    // },
    {
      id: 'evaluation',
      name: 'Evaluation',
      description: 'Assess quality, diversity, and downstream task performance',
      icon: <TrendingUp className="w-6 h-6" />,
      status: 'pending',
      progress: 0,
      component: (
        <Evaluator
          key={`evaluator-${pipelineRunId}-${currentStep}`}
          dataset={pipelineData.dataset}
          autoStart={isRunning && currentStep === 3}
          onComplete={(data) => handleStepComplete(4, data)}
          onLog={(log) =>
            addRequestLog({
              stepId: 'evaluation',
              stepName: 'Evaluation',
              endpoint: '/api/pipeline/evaluator',
              ...log,
            })
          }
        />
      )
    }
  ], [isRunning, currentStep, pipelineData, pipelineRunId]);

  const normalizeExperiencePersonaName = (name: unknown): string => {
    if (typeof name === 'string') return name;
    if (name && typeof name === 'object' && 'name' in (name as object))
      return String((name as { name?: string }).name || 'User');
    return 'User';
  };

  const handleStepComplete = (stepIndex: number, data: unknown) => {
    const newPipelineData = { ...pipelineData };

    switch (stepIndex) {
      case 0: {
        const raw = (data as Experience[]).map((exp) => {
          const personas = (exp.personas && exp.personas.length >= 2)
            ? exp.personas.map((p) => ({
                ...p,
                name: normalizeExperiencePersonaName(p.name),
                traits: Array.isArray(p.traits) ? p.traits : [],
                memory: Array.isArray(p.memory) ? p.memory : []
              }))
            : [
                { role: 'user', name: normalizeExperiencePersonaName((exp as any).user_persona), traits: [], memory: [] },
                { role: 'assistant', name: 'SupportBot', traits: ['helpful', 'professional'], memory: [] as string[] }
              ];
          return {
            ...exp,
            situation: (exp as any).situation ?? (exp as any).context ?? '',
            conversation_starter: (exp as any).conversation_starter ?? (exp as any).first_utterance ?? '',
            goal: exp.goal ?? (exp as any).goal ?? '',
            task: exp.task ?? (exp as any).task ?? (exp as any).goal ?? '',
            constraints: exp.constraints ?? { max_turns: 10, max_tokens_per_turn: 100 },
            personas
          };
        });
        newPipelineData.experiences = raw;
        break;
      }
      case 1:
        newPipelineData.conversations = data as Conversation[];
        break;
      case 2:
        newPipelineData.filteredConversations = data as FilteredConversation[];
        break;
      case 3:
        newPipelineData.dataset = data as DatasetItem[];
        break;
      case 4:
        newPipelineData.evaluations = data as EvaluationMetrics;
        break;
    }

    setPipelineData(newPipelineData);

    if (stepIndex < steps.length - 1) {
      setCurrentStep(stepIndex + 1);
    }
  };

  const runPipeline = async () => {
    console.log('Run Pipeline clicked');
    
    // Validate configuration
    if (selectedDomains.length === 0) {
      alert('Please select at least one domain');
      return;
    }
    if (numDialogues < 1) {
      alert('Number of dialogues must be at least 1');
      return;
    }
    
    // Check backend connection before starting
    if (!backendConnected) {
      const isHealthy = await API_CONFIG.checkHealth();
      if (!isHealthy) {
        const errorMsg = `Backend server is not accessible at ${API_CONFIG.baseUrl}. Please ensure the backend server is running.\n\nTo start the backend:\ncd goalconvo-2\n./start_backend.sh`;
        alert(errorMsg);
        addRequestLog({
          stepId: 'pipeline',
          stepName: 'Pipeline Start',
          endpoint: '/api/run-pipeline',
          status: 'error',
          message: `Backend server not accessible at ${API_CONFIG.baseUrl}`
        });
        return;
      }
      setBackendConnected(true);
    }
    
    // Generate new session ID FIRST (before resetting state)
    const newSessionId = uuidv4();
    sessionIdRef.current = newSessionId;
    
    // Reset state completely
    setRequestLogs([]);
    setPipelineRunId(prev => prev + 1);
    setCurrentStep(0);
    setIsRunning(false); // Ensure running state is reset
    setPipelineData({
      experiences: [],
      conversations: [],
      filteredConversations: [],
      dataset: [],
      evaluations: null
    });
    
    try {
      // Number of dialogues is per domain; total = perDomain × domain count
      const domainCount = selectedDomains.length > 0 ? selectedDomains.length : availableDomains.length;
      const totalDialogues = numDialogues * domainCount;

      // Build request body (ablation overrides)
      const pipelineBody: Record<string, unknown> = {
        num_dialogues: totalDialogues,
        domains: selectedDomains.length > 0 ? selectedDomains : undefined,
        session_id: sessionIdRef.current,
      };
      if (experimentTag.trim()) pipelineBody.experiment_tag = experimentTag.trim();
      if (ablationQualityJudgeOff || ablationFewShot.trim() || ablationTemperature.trim()) {
        const overrides: Record<string, unknown> = {};
        if (ablationQualityJudgeOff) overrides.quality_judge = false;
        const few = parseInt(ablationFewShot.trim(), 10);
        if (ablationFewShot.trim() && !Number.isNaN(few)) overrides.few_shot_examples = few;
        const temp = parseFloat(ablationTemperature.trim());
        if (ablationTemperature.trim() && !Number.isNaN(temp)) overrides.temperature = temp;
        if (Object.keys(overrides).length > 0) pipelineBody.overrides = overrides;
      }
      // Call the new unified pipeline API
      const response = await fetch(API_CONFIG.getUrl(API_CONFIG.endpoints.runPipeline), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(pipelineBody),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `Pipeline start failed: ${response.statusText}`;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || errorMessage;
        } catch {
          errorMessage = errorText || errorMessage;
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      console.log('Pipeline started:', data);
      
      // Join session via WebSocket if connected
      if (socketRef.current?.connected) {
        socketRef.current.emit('join_session', { session_id: sessionIdRef.current });
      } else {
        console.warn('WebSocket not connected, attempting to reconnect...');
        // Try to reconnect if not connected
        if (socketRef.current) {
          socketRef.current.connect();
        }
      }
      
      addRequestLog({
        stepId: 'pipeline',
        stepName: 'Pipeline Start',
        endpoint: '/api/run-pipeline',
        status: 'success',
        message: data.message || 'Pipeline started successfully'
      });
      
      setIsRunning(true);
      setEvaluationStepMessages([]);
    } catch (error) {
      console.error('Failed to start pipeline:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Provide helpful error message for network errors
      let userMessage = errorMessage;
      if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
        userMessage = `Cannot connect to backend server at ${API_CONFIG.baseUrl}. Please ensure:\n1. Backend server is running (cd goalconvo-2 && ./start_backend.sh)\n2. Backend is accessible at ${API_CONFIG.baseUrl}\n3. No firewall is blocking the connection`;
      }
      
      addRequestLog({
        stepId: 'pipeline',
        stepName: 'Pipeline Start',
        endpoint: '/api/run-pipeline',
        status: 'error',
        message: userMessage
      });
      
      alert(`Failed to start pipeline:\n\n${userMessage}`);
    }
  };

  const runEvaluation = async () => {
    if (!backendConnected) {
      const isHealthy = await API_CONFIG.checkHealth();
      if (!isHealthy) {
        alert(`Backend not accessible at ${API_CONFIG.baseUrl}. Please start the backend first.`);
        return;
      }
    }
    setEvaluationStepMessages([]);
    setIsEvaluationRunning(true);
    try {
      // Join session first so we're in the room when backend emits evaluation events
      if (socketRef.current?.connected) {
        socketRef.current.emit('join_session', { session_id: sessionIdRef.current });
        await new Promise((r) => setTimeout(r, 300)); // let server process join before we start evaluation
      }
      const response = await fetch(API_CONFIG.getUrl(API_CONFIG.endpoints.runEvaluation), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          limit: evaluationLimit,
          domains: evaluationDomains.length > 0 ? evaluationDomains : undefined
        })
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || response.statusText);
      }
      addRequestLog({
        stepId: 'evaluation',
        stepName: 'Evaluation',
        endpoint: '/api/run-evaluation',
        status: 'success',
        message: 'Evaluation started (dataset)'
      });
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Unknown error';
      addRequestLog({
        stepId: 'evaluation',
        stepName: 'Evaluation',
        endpoint: '/api/run-evaluation',
        status: 'error',
        message: msg
      });
      setIsEvaluationRunning(false);
      alert(`Failed to start evaluation: ${msg}`);
    }
  };

  /** Same pipeline as dataset evaluation; body includes `dialogues` (validated server-side). */
  const runEvaluationWithUploadedDialogues = async (dialogues: unknown[]) => {
    if (!backendConnected) {
      const isHealthy = await API_CONFIG.checkHealth();
      if (!isHealthy) {
        alert(`Backend not accessible at ${API_CONFIG.baseUrl}. Please start the backend first.`);
        return;
      }
    }
    if (!Array.isArray(dialogues) || dialogues.length === 0) {
      alert('No dialogues to evaluate.');
      return;
    }
    setEvaluationStepMessages([]);
    setIsEvaluationRunning(true);
    try {
      if (socketRef.current?.connected) {
        socketRef.current.emit('join_session', { session_id: sessionIdRef.current });
        await new Promise((r) => setTimeout(r, 300));
      }
      const response = await fetch(API_CONFIG.getUrl(API_CONFIG.endpoints.runEvaluation), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          dialogues
        })
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || response.statusText);
      }
      addRequestLog({
        stepId: 'evaluation',
        stepName: 'Evaluation',
        endpoint: '/api/run-evaluation',
        status: 'success',
        message: `Evaluation started (upload, ${dialogues.length} dialogues)`
      });
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Unknown error';
      addRequestLog({
        stepId: 'evaluation',
        stepName: 'Evaluation',
        endpoint: '/api/run-evaluation',
        status: 'error',
        message: msg
      });
      setIsEvaluationRunning(false);
      alert(`Failed to start evaluation: ${msg}`);
    }
  };

  const extractDialoguesFromUploadedJson = (parsed: unknown): unknown[] => {
    if (Array.isArray(parsed)) return parsed;
    if (parsed && typeof parsed === 'object' && 'dialogues' in parsed) {
      const d = (parsed as { dialogues: unknown }).dialogues;
      if (Array.isArray(d)) return d;
    }
    throw new Error('JSON must be a dialogue array or an object with a "dialogues" array (same shape as pipeline export).');
  };

  const onEvaluationJsonFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = '';
    if (files.length === 0) return;
    try {
      const parsedDialoguesPerFile = await Promise.all(
        files.map(async (file) => {
          const text = await file.text();
          const parsed = JSON.parse(text) as unknown;
          return extractDialoguesFromUploadedJson(parsed);
        })
      );
      const mergedDialogues = parsedDialoguesPerFile.flat();
      setUploadedEvaluationDialogues((prev) => [...prev, ...mergedDialogues]);
      setUploadedEvaluationFileNames((prev) => [...prev, ...files.map((file) => file.name)]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Invalid JSON file';
      alert(msg);
    }
  };

  const resetPipeline = () => {
    console.log('Reset Pipeline clicked');
    setIsRunning(false);
    setIsEvaluationRunning(false);
    setCurrentStep(0);
    setPipelineRunId(0);
    setRequestLogs([]);
    setEvaluationDomains([]);
    setEvaluationLimit(10);
    setUploadedEvaluationDialogues([]);
    setUploadedEvaluationFileNames([]);
    setPipelineData({
      experiences: [],
      conversations: [],
      filteredConversations: [],
      dataset: [],
      evaluations: {
        overall_score: 0,
        diversity_score: 0,
        coherence_score: 0,
        task_success_rate: 0,
        fluency_score: 0,
        groundedness_score: 0,
        categories: {
          lexical_diversity: 0,
          conversation_length: { avg_turns: 0, std_dev: 0 },
          domain_distribution: {},
          task_success_by_domain: {},
          goal_completion_by_domain: {}
        }
      }
    });
  };

  const downloadDialoguesJSON = () => {
    if (pipelineData.conversations.length === 0) {
      alert('No dialogues available to download');
      return;
    }

    // Prepare dialogues data for download
    const dialoguesData = {
      metadata: {
        generated_at: new Date().toISOString(),
        total_dialogues: pipelineData.conversations.length,
        total_turns: pipelineData.conversations.reduce((sum, conv) => sum + (conv.turns?.length || 0), 0),
        domains: [...new Set(pipelineData.conversations.map(c => c.domain))],
        pipeline_run_id: pipelineRunId
      },
      dialogues: pipelineData.conversations.map(conv => ({
        dialogue_id: conv.conv_id,
        domain: conv.domain,
        goal: conv.goal || conv.task,
        context: conv.situation,
        user_persona: conv.personas?.find(p => p.role === 'user')?.name || 'General user',
        turns: conv.turns?.map(turn => ({
          role: turn.speaker === 'user' ? 'User' : 'SupportBot',
          text: turn.text,
          turn_id: turn.turn_id
        })) || [],
        metadata: {
          num_turns: conv.turns?.length || 0,
          task_success: conv.task_success,
          quality_score: conv.judge_score ? conv.judge_score / 5 : 0,
          generated_at: conv.provenance?.timestamp || new Date().toISOString(),
          model_version: conv.provenance?.generator_model || 'unknown'
        }
      }))
    };

    // Create and download JSON file
    const jsonString = JSON.stringify(dialoguesData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `goalconvo-dialogues-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const domainPillColors: Record<string, string> = {
    hotel: 'bg-orange-500/30 border-orange-400/60 text-orange-200 hover:bg-orange-500/40',
    restaurant: 'bg-fuchsia-500/30 border-fuchsia-400/60 text-fuchsia-200 hover:bg-fuchsia-500/40',
    taxi: 'bg-sky-500/30 border-sky-400/60 text-sky-200 hover:bg-sky-500/40',
    train: 'bg-lime-500/25 border-lime-400/60 text-lime-200 hover:bg-lime-500/35',
    attraction: 'bg-indigo-500/30 border-indigo-400/60 text-indigo-200 hover:bg-indigo-500/40',
  };

  return (
    <div className="app-canvas max-w-7xl mx-auto px-4 py-6">
      {/* Top-level tabs + Light/Dark toggle */}
      <div className="flex items-center gap-4 mb-8 flex-wrap">
        <div className="flex gap-1 p-1.5 bg-white/5 rounded-2xl border border-white/10 w-fit shadow-lg shadow-black/20 dashboard-tabs">
          <button
            onClick={() => setActiveTab('pipeline')}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold transition-all duration-200 ${
              activeTab === 'pipeline'
                ? 'bg-gradient-to-r from-sky-500 via-indigo-500 to-fuchsia-500 text-white shadow-lg shadow-fuchsia-500/30'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <Play className="w-4 h-4" /> Pipeline
          </button>
          <button
            onClick={() => setActiveTab('versions')}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold transition-all duration-200 ${
              activeTab === 'versions'
                ? 'bg-gradient-to-r from-fuchsia-500 to-violet-600 text-white shadow-lg shadow-fuchsia-500/30'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <GitBranch className="w-4 h-4" /> Versions
          </button>
          <button
            onClick={() => setActiveTab('human-eval')}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold transition-all duration-200 ${
              activeTab === 'human-eval'
                ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white shadow-lg shadow-orange-500/30'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <ClipboardList className="w-4 h-4" /> Human Evaluation
          </button>
        </div>
        <button
          type="button"
          onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
          className="flex items-center justify-center w-10 h-10 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors dashboard-tabs"
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun className="w-5 h-5 text-amber-400" /> : <Moon className="w-5 h-5 text-indigo-400" />}
        </button>
      </div>

      {activeTab === 'versions' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4">
          <Versions />
        </motion.div>
      )}

      {activeTab === 'human-eval' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4">
          <HumanEvaluation />
        </motion.div>
      )}

      {activeTab === 'pipeline' && (
        <>
      {/* Control Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="dashboard-control relative bg-gradient-to-br from-white/15 to-white/5 backdrop-blur-xl rounded-2xl p-6 mb-8 border border-white/20 shadow-xl shadow-black/20 overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-orange-500/5 via-fuchsia-500/5 to-sky-500/5 pointer-events-none" />
        <div className="relative">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-orange-300 via-fuchsia-300 to-sky-300 bg-clip-text text-transparent mb-2">Pipeline Control</h2>
            <p className="text-gray-300">Manage the entire GoalConvo generation pipeline</p>
            <div className="flex items-center gap-2 mt-2">
              <div className={`w-2.5 h-2.5 rounded-full ${backendConnected ? 'bg-lime-400 shadow-[0_0_8px_rgba(163,230,53,0.7)]' : 'bg-rose-400 shadow-[0_0_8px_rgba(251,113,133,0.6)]'} ${backendCheckInProgress ? 'animate-pulse' : ''}`} />
              <span className="text-xs text-gray-400">
                {backendCheckInProgress ? 'Checking...' : backendConnected ? 'Backend Connected' : 'Backend Disconnected'}
              </span>
              {!backendConnected && (
                <span className="text-xs text-orange-300 ml-2">
                  ({API_CONFIG.baseUrl})
                </span>
              )}
            </div>
          </div>
          <div className="flex gap-3">
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
              onClick={runPipeline}
              disabled={isRunning || !backendConnected}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-fuchsia-500 via-indigo-500 to-sky-500 text-white font-semibold rounded-xl shadow-lg shadow-fuchsia-500/35 hover:shadow-fuchsia-500/45 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              title={!backendConnected ? 'Backend server is not connected. Please start the backend server first.' : ''}
            >
              <Play className="w-5 h-5" />
              {isRunning ? 'Running...' : 'Run Pipeline'}
            </motion.button>
            {pipelineData.conversations.length > 0 && (
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
                onClick={downloadDialoguesJSON}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-lime-500 to-green-500 text-white font-semibold rounded-xl shadow-lg shadow-lime-500/35 hover:shadow-lime-500/45 transition-all"
                title={`Download ${pipelineData.conversations.length} dialogue(s) as JSON`}
              >
                <Download className="w-5 h-5" />
                Download Dialogues ({pipelineData.conversations.length})
              </motion.button>
            )}
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
              onClick={resetPipeline}
              className="flex items-center gap-2 px-6 py-3 bg-white/10 text-white font-semibold rounded-xl hover:bg-white/20 transition-all border border-white/20"
            >
              <RotateCcw className="w-5 h-5" />
              Reset
            </motion.button>
          </div>
        </div>

        {/* Pipeline Configuration */}
        <div className='grid grid-cols-1 md:grid-cols-2 gap-4 '>

        <div className="dashboard-input-wrap grid grid-cols-1 gap-4 mb-6 p-5 bg-white/5 rounded-xl border border-white/10 shadow-inner">
          <div>
            <label className="block text-sm font-semibold text-sky-200 mb-2">
              Dialogues per domain
            </label>
            <div className="flex gap-2">
              <input
                type="number"
                min="1"
                max="100"
                value={numDialogues}
                onChange={(e) => setNumDialogues(Math.max(1, parseInt(e.target.value) || 1))}
                disabled={isRunning}
                className="flex-1 px-4 py-2.5 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-fuchsia-400 focus:border-fuchsia-400/50 disabled:opacity-50 disabled:cursor-not-allowed"
                placeholder="Per domain"
              />
            </div>
            <p className="text-xs text-gray-400 mt-1.5">
              {numDialogues} per domain → <span className="text-fuchsia-300 font-medium">{numDialogues * (selectedDomains.length || availableDomains.length)} total</span>
              {(selectedDomains.length || availableDomains.length) > 0 && ` across ${selectedDomains.length || availableDomains.length} domain(s)`}
            </p>
          </div>
          <div>
            <label className="block text-sm font-semibold text-fuchsia-200 mb-2">
              Select Domains
            </label>
            <div className="flex flex-wrap gap-2">
              {availableDomains.map((domain) => (
                <label
                  key={domain}
                  className={`flex items-center gap-2 px-3 py-2 rounded-xl border cursor-pointer transition-all ${
                    selectedDomains.includes(domain)
                      ? domainPillColors[domain] ?? 'bg-fuchsia-500/25 border-fuchsia-400 text-white'
                      : 'bg-white/5 border-white/20 text-gray-400 hover:bg-white/10 hover:text-gray-200'
                  } ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <input
                    type="checkbox"
                    checked={selectedDomains.includes(domain)}
                    onChange={(e) => {
                      if (isRunning) return;
                      if (e.target.checked) {
                        setSelectedDomains([...selectedDomains, domain]);
                      } else {
                        setSelectedDomains(selectedDomains.filter(d => d !== domain));
                      }
                    }}
                    disabled={isRunning}
                    className="w-4 h-4 rounded focus:ring-2 focus:ring-offset-0 focus:ring-fuchsia-400 accent-fuchsia-500"
                  />
                  <span className="text-sm font-medium capitalize">{domain}</span>
                </label>
              ))}
            </div>
            {selectedDomains.length === 0 && (
              <p className="text-xs text-orange-300 mt-2">⚠️ Please select at least one domain</p>
            )}
          </div>
           {/* Request Log Stack */}
            </div>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="dashboard-request-log bg-white/5 rounded-2xl border border-white/10 p-4 flex flex-col max-h-[13.8rem] shadow-inner"
        >
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-lg font-semibold text-white">Request Log</h3>
              <p className="text-xs text-gray-400">Latest API calls across the pipeline</p>
            </div>
            <span className="text-xs text-gray-300 bg-white/10 px-3 py-1.5 rounded-full border border-white/10">
              {requestLogs.length} events
            </span>
          </div>

          <div className="mt-2 space-y-2 overflow-y-auto">
            {requestLogs.length === 0 ? (
              <p className="text-xs text-gray-400">
                No requests yet. Run the pipeline to see activity.
              </p>
            ) : (
              requestLogs.map((log) => (
                <div
                  key={log.id}
                  className={`rounded-xl border px-3 py-2 text-xs flex flex-col gap-1 ${
                    log.status === 'success'
                      ? 'border-lime-400/50 bg-gradient-to-r from-lime-500/20 to-green-500/10'
                      : 'border-rose-400/50 bg-gradient-to-r from-rose-500/20 to-pink-500/10'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-semibold text-white">
                      {log.stepName}
                    </span>
                    <span className="text-[10px] text-gray-300">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[11px] text-sky-200">
                      {log.endpoint}
                    </span>
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                        log.status === 'success'
                        ? 'bg-lime-500/40 text-lime-100'
                        : 'bg-rose-500/40 text-rose-100'
                      }`}
                      >
                      {log.status.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-[11px] text-gray-100 line-clamp-2">
                    {log.message}
                  </p>
                </div>
              ))
            )}
          </div>
        </motion.div>
        </div>

        {/* Experiment / ablation options (for tagging and comparing versions) */}
        {/* <div className="mb-6 p-4 bg-white/5 rounded-xl border border-white/10">
          <h4 className="text-sm font-semibold text-gray-300 mb-3">Experiment / ablation (optional)</h4>
          <p className="text-xs text-gray-400 mb-3">Use these to run ablations (e.g. no quality judge, few-shot=0, different temperature) and tag versions for comparison in the Versions tab.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Experiment tag</label>
              <input
                type="text"
                value={experimentTag}
                onChange={(e) => setExperimentTag(e.target.value)}
                disabled={isRunning}
                placeholder="e.g. ablation-no-judge"
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={ablationQualityJudgeOff}
                  onChange={(e) => setAblationQualityJudgeOff(e.target.checked)}
                  disabled={isRunning}
                  className="w-4 h-4 text-cyan-500 rounded focus:ring-cyan-500"
                />
                <span className="text-sm text-gray-300">Disable quality judge</span>
              </label>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Few-shot examples (blank = default)</label>
              <input
                type="text"
                value={ablationFewShot}
                onChange={(e) => setAblationFewShot(e.target.value)}
                disabled={isRunning}
                placeholder="e.g. 0"
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Temperature (blank = default)</label>
              <input
                type="text"
                value={ablationTemperature}
                onChange={(e) => setAblationTemperature(e.target.value)}
                disabled={isRunning}
                placeholder="e.g. 0.5"
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
            </div>
          </div>
        </div> */}

        {/* Progress Overview - step boxes with distinct colors per step */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          {steps.map((step, index) => {
            const isActive = isRunning && index === currentStep;
            const stepAccents = [
              { border: 'orange', bg: 'orange-500/20', icon: 'text-orange-400' },
              { border: 'fuchsia', bg: 'fuchsia-500/20', icon: 'text-fuchsia-400' },
              { border: 'rose', bg: 'rose-500/20', icon: 'text-rose-400' },
              { border: 'lime', bg: 'lime-500/20', icon: 'text-lime-400' },
            ];
            const accent = stepAccents[index % stepAccents.length];
            const gradient = index === 0
              ? 'conic-gradient(from 0deg, #fb923c, #d946ef, #38bdf8, #a3e635, #fb923c)'
              : index === 1
              ? 'conic-gradient(from 0deg, #d946ef, #818cf8, #a3e635, #38bdf8, #d946ef)'
              : index === 2
              ? 'conic-gradient(from 0deg, #fb7185, #818cf8, #facc15, #a3e635, #fb7185)'
              : 'conic-gradient(from 0deg, #a3e635, #38bdf8, #d946ef, #fb923c, #a3e635)';
            const glowColor = index === 0 ? 'rgba(251, 146, 60, 0.45)' : index === 1 ? 'rgba(217, 70, 239, 0.45)' : index === 2 ? 'rgba(251, 113, 133, 0.45)' : 'rgba(163, 230, 53, 0.45)';
            const glowColorSoft = index === 0 ? 'rgba(251, 146, 60, 0.15)' : index === 1 ? 'rgba(217, 70, 239, 0.15)' : index === 2 ? 'rgba(251, 113, 133, 0.15)' : 'rgba(163, 230, 53, 0.15)';
            const boxContent = (
              <>
                <div className={`flex items-center gap-2 mb-2 text-xl ${!isActive ? accent.icon : ''}`}>
                  {step.icon}
                  <span className="text-lg font-semibold text-white">{step.name}</span>
                </div>
                <div className="w-full bg-white/20 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${step.progress}%` }}
                    className="h-2 rounded-full transition-all"
                    style={{
                      backgroundColor: index < currentStep
                        ? 'rgb(163 230 53)'
                        : index === 0 ? 'rgb(251 146 60 / 0.9)' : index === 1 ? 'rgb(217 70 239 / 0.9)' : index === 2 ? 'rgb(251 113 133 / 0.9)' : 'rgb(163 230 53 / 0.9)'
                    }}
                  />
                </div>
              </>
            );
            return (
              <motion.div
                key={step.id}
                initial={{ opacity: 0.5 }}
                animate={{
                  opacity: index <= currentStep ? 1 : 0.6,
                  scale: isActive ? 1.03 : 1
                }}
                className={
                  isActive
                    ? 'relative rounded-xl p-[4px] overflow-hidden min-h-[80px]'
                    : `p-3 rounded-xl border transition-all ${
                        index < currentStep
                          ? 'border-lime-400 bg-lime-500/20'
                          : 'border-white/20 bg-white/5 hover:border-white/30'
                      }`
                }
              >
                {isActive ? (
                  <>
                    <div
                      className="step-border-loader absolute inset-0 rounded-xl"
                      style={{
                        background: gradient,
                        boxShadow: `0 0 20px ${glowColor}, 0 0 40px ${glowColorSoft}`,
                      }}
                    />
                    <div className="relative m-[4px] rounded-[10px] bg-slate-900/95 p-3 min-h-[80px] flex flex-col backdrop-blur-sm">
                      {boxContent}
                    </div>
                  </>
                ) : (
                  boxContent
                )}
              </motion.div>
            );
          })}
        </div>
        </div>
      </motion.div>

      {/* Live dialogue viewer (during simulation) */}
      {isRunning && liveDialogue && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-gradient-to-br from-fuchsia-500/10 to-sky-500/10 backdrop-blur-md rounded-2xl p-6 mb-8 border border-fuchsia-400/40 shadow-lg shadow-fuchsia-500/20"
        >
          <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-fuchsia-400 animate-pulse shadow-[0_0_8px_rgba(217,70,239,0.8)]" />
            Live dialogue
            {liveDialogue.dialogue_index != null && liveDialogue.total_dialogues != null && (
              <span className="text-gray-400 font-normal">
                (dialogue {liveDialogue.dialogue_index} of {liveDialogue.total_dialogues})
              </span>
            )}
          </h3>
          <p className="text-sm text-fuchsia-200 mb-4">{liveDialogue.step_message}</p>
          {liveDialogue.goal && (
            <p className="text-xs text-gray-400 mb-3 truncate" title={liveDialogue.goal}>Goal: {liveDialogue.goal}</p>
          )}
          <div className="space-y-2 max-h-64 overflow-y-auto rounded-xl bg-black/25 p-3 border border-white/5">
            {liveDialogue.current_turns.map((turn: { role: string; text: string }, idx: number) => (
              <div
                key={idx}
                className={`text-sm p-3 rounded-xl ${
                  turn.role === 'User'
                    ? 'bg-gradient-to-r from-sky-500/25 to-indigo-500/20 text-left border-l-2 border-sky-400'
                    : 'bg-gradient-to-r from-fuchsia-500/25 to-violet-500/20 text-left ml-4 border-l-2 border-fuchsia-400'
                }`}
              >
                <span className="font-medium text-gray-300">{turn.role}: </span>
                <span className="text-white">{turn.text || ''}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Evaluation: separate step after dialogue generation; show Run Evaluation button when dialogues exist */}
      <div className="grid grid-cols-1 gap-6">
        <AnimatePresence mode="wait">
          {!isRunning ? (
            <motion.div
              key="evaluation-panel"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.5 }}
              className="bg-gradient-to-br from-white/15 to-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/20 shadow-xl lg:col-span-2"
            >
              {isEvaluationRunning ? (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 text-fuchsia-300">
                    <span className="inline-block w-2 h-2 rounded-full bg-fuchsia-400 animate-pulse" />
                    <span className="font-medium">Evaluation in progress</span>
                  </div>
                  <div className="rounded-xl bg-black/20 border border-white/10 overflow-hidden">
                    <div className="px-4 py-2 text-xs font-medium text-gray-400 border-b border-white/10 bg-white/5">
                      Current step
                    </div>
                    <ul className="max-h-64 overflow-y-auto py-2 divide-y divide-white/5">
                      {evaluationStepMessages.length === 0 ? (
                        <li className="px-4 py-2 text-gray-500 text-sm">Starting evaluation…</li>
                      ) : (
                        evaluationStepMessages.map((msg, i) => (
                          <li key={i} className="px-4 py-2 text-sm text-gray-300 flex items-center gap-2">
                            {i === evaluationStepMessages.length - 1 ? (
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-fuchsia-400 animate-pulse" />
                            ) : (
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-500/60" />
                            )}
                            {msg}
                          </li>
                        ))
                      )}
                    </ul>
                  </div>
                </div>
              ) : (
                <>
                  <div className="mb-6 rounded-xl border border-indigo-400/25 bg-indigo-500/10 p-4 text-left">
                    <div className="flex flex-wrap items-center gap-3 mb-2">
                      <Upload className="w-5 h-5 text-indigo-300 shrink-0" />
                      <h4 className="text-sm font-semibold text-white">Evaluate from dialogues JSON</h4>
                    </div>
                    <p className="text-xs text-gray-400 mb-3">
                      Upload one or more files produced by this app’s export (or matching the schema below). Uses the same
                      comprehensive evaluation as “Run Evaluation” on the dataset.
                    </p>
                    <label className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600/80 hover:bg-indigo-500 text-white text-sm font-medium cursor-pointer border border-indigo-400/40">
                      <Upload className="w-4 h-4" />
                      Choose JSON files
                      <input
                        type="file"
                        accept=".json,application/json"
                        multiple
                        className="hidden"
                        onChange={onEvaluationJsonFileSelected}
                      />
                    </label>
                    {uploadedEvaluationFileNames.length > 0 && (
                      <div className="mt-3">
                        <p className="text-xs text-indigo-100 mb-3">
                          Loaded {uploadedEvaluationFileNames.length} file(s), {uploadedEvaluationDialogues.length} dialogue(s).
                        </p>
                        <button
                          type="button"
                          onClick={() => runEvaluationWithUploadedDialogues(uploadedEvaluationDialogues)}
                          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-fuchsia-500 via-indigo-500 to-sky-500 text-white text-sm font-semibold shadow-lg shadow-fuchsia-500/25 hover:shadow-fuchsia-500/40 transition-all"
                        >
                          <Play className="w-4 h-4" />
                          Run Evaluation (Uploaded JSON)
                        </button>
                      </div>
                    )}
                    <details className="mt-4 group">
                      <summary className="text-xs text-indigo-200 cursor-pointer hover:text-white">
                        Expected JSON schema (evaluation pipeline)
                      </summary>
                      <pre className="mt-2 p-3 rounded-lg bg-black/40 border border-white/10 text-[11px] text-gray-300 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
                        {EVALUATION_DIALOGUES_JSON_SCHEMA}
                      </pre>
                    </details>
                  </div>

                  {pipelineData.evaluations != null && typeof pipelineData.evaluations === 'object' ? (
                <Evaluator
                  key={`evaluator-${pipelineRunId}`}
                  dataset={pipelineData.dataset}
                  metrics={pipelineData.evaluations}
                  autoStart={false}
                  onComplete={(metrics) => {
                    console.log('Evaluation complete:', metrics);
                  }}
                  onLog={(log) =>
                    addRequestLog({
                      stepId: 'evaluation',
                      stepName: 'Evaluation',
                      endpoint: '/api/run-evaluation',
                      status: log.status,
                      message: log.message
                    })
                  }
                />
              ) : pipelineData.conversations.length > 0 ? (
                <div className="flex flex-col items-center justify-center py-14 text-center">
                  <div className="p-5 rounded-2xl bg-gradient-to-br from-fuchsia-500/30 via-indigo-500/20 to-lime-500/20 mb-5 shadow-lg border border-white/10">
                    <TrendingUp className="w-14 h-14 text-fuchsia-300" />
                  </div>
                  <h3 className="text-xl font-bold bg-gradient-to-r from-fuchsia-200 via-indigo-200 to-lime-200 bg-clip-text text-transparent mb-2">Evaluation</h3>
                  <p className="text-gray-400 max-w-md mb-6">
                    Dialogue generation is done. Run evaluation to see quality and diversity metrics.
                  </p>
                  <div className="w-full max-w-xl mb-6 space-y-4">
                    <div className="text-left">
                      <div className="text-xs font-semibold text-gray-400 mb-2">Domains (optional)</div>
                      <div className="flex flex-wrap gap-2">
                        {availableDomains.map((d) => {
                          const active = evaluationDomains.includes(d);
                          return (
                            <button
                              key={d}
                              type="button"
                              onClick={() =>
                                setEvaluationDomains((prev) =>
                                  prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]
                                )
                              }
                              className={`px-3 py-1.5 rounded-full text-sm font-semibold border transition-all ${
                                active
                                  ? 'bg-white/15 border-white/40 text-white'
                                  : 'bg-white/5 border-white/15 text-gray-300 hover:bg-white/10'
                              }`}
                            >
                              {d}
                            </button>
                          );
                        })}
                        {evaluationDomains.length > 0 && (
                          <button
                            type="button"
                            onClick={() => setEvaluationDomains([])}
                            className="px-3 py-1.5 rounded-full text-sm font-semibold border bg-white/5 border-white/15 text-gray-300 hover:bg-white/10"
                          >
                            Clear
                          </button>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        If none selected, evaluation uses latest dialogues across all domains.
                      </div>
                    </div>
                    <div className="text-left">
                      <div className="text-xs font-semibold text-gray-400 mb-2">Number of dialogues</div>
                      <input
                        type="number"
                        min={1}
                        max={500}
                        value={evaluationLimit}
                        onChange={(e) => setEvaluationLimit(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
                        className="w-40 px-3 py-2 rounded-xl bg-black/20 border border-white/15 text-white focus:outline-none focus:border-white/30"
                      />
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={runEvaluation}
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold bg-gradient-to-r from-fuchsia-500 via-indigo-500 to-sky-500 text-white shadow-lg shadow-fuchsia-500/30 hover:shadow-fuchsia-500/50 transition-all duration-200 disabled:opacity-50"
                  >
                    <Play className="w-5 h-5" />
                    Run Evaluation
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-14 text-center">
                  <div className="p-5 rounded-2xl bg-gradient-to-br from-fuchsia-500/30 via-indigo-500/20 to-lime-500/20 mb-5 shadow-lg border border-white/10">
                    <TrendingUp className="w-14 h-14 text-fuchsia-300" />
                  </div>
                  <h3 className="text-xl font-bold bg-gradient-to-r from-fuchsia-200 via-indigo-200 to-lime-200 bg-clip-text text-transparent mb-2">Evaluation</h3>
                  <p className="text-gray-400 max-w-md mb-6">
                    Run evaluation on the latest dialogues from the dataset. Generate dialogues first if none exist yet.
                  </p>
                  <div className="w-full max-w-xl mb-6 space-y-4">
                    <div className="text-left">
                      <div className="text-xs font-semibold text-gray-400 mb-2">Domains (optional)</div>
                      <div className="flex flex-wrap gap-2">
                        {availableDomains.map((d) => {
                          const active = evaluationDomains.includes(d);
                          return (
                            <button
                              key={d}
                              type="button"
                              onClick={() =>
                                setEvaluationDomains((prev) =>
                                  prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]
                                )
                              }
                              className={`px-3 py-1.5 rounded-full text-sm font-semibold border transition-all ${
                                active
                                  ? 'bg-white/15 border-white/40 text-white'
                                  : 'bg-white/5 border-white/15 text-gray-300 hover:bg-white/10'
                              }`}
                            >
                              {d}
                            </button>
                          );
                        })}
                        {evaluationDomains.length > 0 && (
                          <button
                            type="button"
                            onClick={() => setEvaluationDomains([])}
                            className="px-3 py-1.5 rounded-full text-sm font-semibold border bg-white/5 border-white/15 text-gray-300 hover:bg-white/10"
                          >
                            Clear
                          </button>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        If none selected, evaluation uses latest dialogues across all domains.
                      </div>
                    </div>
                    <div className="text-left">
                      <div className="text-xs font-semibold text-gray-400 mb-2">Number of dialogues</div>
                      <input
                        type="number"
                        min={1}
                        max={500}
                        value={evaluationLimit}
                        onChange={(e) => setEvaluationLimit(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
                        className="w-40 px-3 py-2 rounded-xl bg-black/20 border border-white/15 text-white focus:outline-none focus:border-white/30"
                      />
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={runEvaluation}
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold bg-gradient-to-r from-fuchsia-500 via-indigo-500 to-sky-500 text-white shadow-lg shadow-fuchsia-500/30 hover:shadow-fuchsia-500/50 transition-all duration-200 disabled:opacity-50"
                  >
                    <Play className="w-5 h-5" />
                    Run Evaluation
                  </button>
                </div>
              )}
                </>
              )}
            </motion.div>
          ) : (
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.5 }}
              className="bg-gradient-to-br from-white/15 to-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/20 shadow-xl lg:col-span-2"
            >
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-gradient-to-br from-fuchsia-500 via-indigo-500 to-sky-500 rounded-xl shadow-lg">
                  {steps[Math.min(currentStep, steps.length - 1)]?.icon}
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-white mb-1">
                    {steps[Math.min(currentStep, steps.length - 1)]?.name}
                  </h3>
                  <p className="text-gray-300">{steps[Math.min(currentStep, steps.length - 1)]?.description}</p>
                </div>
              </div>

              {currentStep === 3 ? (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 text-fuchsia-300">
                    <span className="inline-block w-2 h-2 rounded-full bg-fuchsia-400 animate-pulse" />
                    <span className="font-medium">Evaluation in progress</span>
                  </div>
                  <div className="rounded-xl bg-black/20 border border-white/10 overflow-hidden">
                    <div className="px-4 py-2 text-xs font-medium text-gray-400 border-b border-white/10 bg-white/5">
                      Current step
                    </div>
                    <ul className="max-h-64 overflow-y-auto py-2 divide-y divide-white/5">
                      {evaluationStepMessages.length === 0 ? (
                        <li className="px-4 py-2 text-gray-500 text-sm">Waiting for evaluation steps…</li>
                      ) : (
                        evaluationStepMessages.map((msg, i) => (
                          <li key={i} className="px-4 py-2 text-sm text-gray-300 flex items-center gap-2">
                            {i === evaluationStepMessages.length - 1 ? (
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-fuchsia-400 animate-pulse" />
                            ) : (
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-500/60" />
                            )}
                            {msg}
                          </li>
                        ))
                      )}
                    </ul>
                  </div>
                </div>
              ) : (
                steps[Math.min(currentStep, steps.length - 1)]?.component ?? null
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Pipeline Data Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4"
      >
        <div className="bg-gradient-to-br from-amber-500/15 to-orange-500/10 backdrop-blur-md rounded-xl p-5 border border-amber-400/20 card-hover">
          <div className="text-amber-300 font-semibold mb-1">Experiences</div>
          <div className="text-3xl font-bold text-white">{pipelineData.experiences.length}</div>
        </div>
        <div className="bg-gradient-to-br from-orange-500/15 to-sky-500/10 backdrop-blur-md rounded-xl p-5 border border-orange-400/20 card-hover">
          <div className="text-sky-300 font-semibold mb-1">Conversations</div>
          <div className="text-3xl font-bold text-white">{pipelineData.conversations.length}</div>
        </div>
        <div className="bg-gradient-to-br from-rose-500/15 to-pink-500/10 backdrop-blur-md rounded-xl p-5 border border-rose-400/20 card-hover">
          <div className="text-rose-300 font-semibold mb-1">Filtered</div>
          <div className="text-3xl font-bold text-white">{pipelineData.filteredConversations.length}</div>
        </div>
        <div className="bg-gradient-to-br from-lime-500/15 to-green-500/10 backdrop-blur-md rounded-xl p-5 border border-lime-400/20 card-hover">
          <div className="text-lime-300 font-semibold mb-1">Dataset Size</div>
          <div className="text-3xl font-bold text-white">{pipelineData.dataset.length}</div>
        </div>
      </motion.div>

      {/* Generated Experiences and Dialogues Display - Shown when pipeline completes */}
      {!isRunning && (pipelineData.experiences.length > 0 || pipelineData.conversations.length > 0) && (
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1, duration: 0.5 }}
          className="mt-8 space-y-6"
        >
          {/* Generated Experiences Section */}
          {pipelineData.experiences.length > 0 && (
            <div className="bg-gradient-to-br from-orange-500/15 via-fuchsia-500/10 to-sky-500/15 backdrop-blur-md rounded-2xl p-6 border border-orange-400/30 shadow-xl">
              <div className="flex items-center gap-3 mb-4">
                <Brain className="w-6 h-6 text-orange-400" />
                <h2 className="text-2xl font-bold bg-gradient-to-r from-orange-200 via-fuchsia-200 to-sky-200 bg-clip-text text-transparent">Generated Experiences</h2>
                <span className="ml-auto px-3 py-1.5 bg-orange-500/30 text-orange-100 rounded-full text-sm font-semibold border border-orange-400/30">
                  {pipelineData.experiences.length} {pipelineData.experiences.length === 1 ? 'Experience' : 'Experiences'}
                </span>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {pipelineData.experiences.map((experience, idx) => (
                  <motion.div
                    key={experience.experience_id || idx}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: idx * 0.1 }}
                    className="bg-white/10 backdrop-blur-sm rounded-xl p-5 border border-white/20 hover:border-amber-400/50 transition-all card-hover"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="text-sm font-semibold text-fuchsia-400 mb-1">
                          {experience.domain.toUpperCase()}
                        </div>
                        <h3 className="text-lg font-bold text-white mb-2">
                          {experience.task || experience.goal}
                        </h3>
                      </div>
                      <div className="text-xs text-gray-400 bg-white/10 px-2 py-1 rounded">
                        #{idx + 1}
                      </div>
                    </div>
                    
                    <div className="space-y-2 text-sm">
                      <div>
                        <span className="text-gray-400">Goal:</span>
                        <p className="text-white mt-1">{experience.goal}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">Situation:</span>
                        <p className="text-white/90 mt-1 line-clamp-2">{experience.situation}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">Starter:</span>
                        <p className="text-sky-200 mt-1 italic">"{experience.conversation_starter}"</p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* Generated Dialogues Section */}
          {pipelineData.conversations.length > 0 && (
            <div className="bg-gradient-to-br from-fuchsia-500/15 via-indigo-500/10 to-rose-500/15 backdrop-blur-md rounded-2xl p-6 border border-fuchsia-400/30 shadow-xl">
              <div className="flex items-center gap-3 mb-4">
                <Users className="w-6 h-6 text-fuchsia-400" />
                <h2 className="text-2xl font-bold bg-gradient-to-r from-fuchsia-200 via-indigo-200 to-rose-200 bg-clip-text text-transparent">Generated Dialogues</h2>
                <span className="ml-auto px-3 py-1.5 bg-fuchsia-500/30 text-fuchsia-100 rounded-full text-sm font-semibold border border-fuchsia-400/30">
                  {pipelineData.conversations.length} {pipelineData.conversations.length === 1 ? 'Dialogue' : 'Dialogues'}
                </span>
              </div>
              
              <div className="space-y-4">
                {pipelineData.conversations.map((conversation, idx) => (
                  <motion.div
                    key={conversation.conv_id || idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="bg-white/10 backdrop-blur-sm rounded-xl p-5 border border-white/20 hover:border-fuchsia-400/50 transition-all card-hover"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-sm font-semibold text-fuchsia-400 bg-fuchsia-500/20 px-3 py-1 rounded-full">
                            {conversation.domain.toUpperCase()}
                          </span>
                          <span className="text-xs text-gray-400">
                            {conversation.turns?.length || 0} turns
                          </span>
                          {conversation.task_success && (
                            <span className="text-xs bg-green-500/30 text-green-100 px-2 py-1 rounded-full">
                              ✓ Task Success
                            </span>
                          )}
                        </div>
                        <h3 className="text-lg font-bold text-white mb-1">
                          {conversation.task || conversation.goal}
                        </h3>
                        {conversation.judge_score && (
                          <div className="flex items-center gap-2 mt-2">
                            <span className="text-xs text-gray-400">Quality Score:</span>
                            <div className="flex-1 bg-white/20 rounded-full h-2 max-w-[200px]">
                              <div
                                className="bg-gradient-to-r from-fuchsia-400 to-pink-400 h-2 rounded-full"
                                style={{ width: `${(conversation.judge_score / 5) * 100}%` }}
                              />
                            </div>
                            <span className="text-xs text-white font-semibold">
                              {conversation.judge_score?.toFixed(1)}/5.0
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Conversation Preview */}
                    {conversation.turns && conversation.turns.length > 0 && (
                      <div className="mt-4 space-y-2 max-h-60 overflow-y-auto">
                        {conversation.turns.slice(0, 4).map((turn, turnIdx) => (
                          <div
                            key={turnIdx}
                            className={`flex gap-3 p-2 rounded-lg ${
                              turn.speaker === 'user'
                                ? 'bg-sky-500/10 border-l-2 border-sky-400'
                                : 'bg-fuchsia-500/10 border-l-2 border-fuchsia-400'
                            }`}
                          >
                            <span className={`text-xs font-semibold min-w-[60px] ${
                              turn.speaker === 'user' ? 'text-sky-300' : 'text-fuchsia-300'
                            }`}>
                              {turn.speaker === 'user' ? '👤 User' : '🤖 Bot'}
                            </span>
                            <p className="text-sm text-white/90 flex-1">{turn.text}</p>
                          </div>
                        ))}
                        {conversation.turns.length > 4 && (
                          <div className="text-center text-xs text-gray-400 pt-2">
                            + {conversation.turns.length - 4} more turns
                          </div>
                        )}
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}
        </>
      )}
    </div>
  );
}
