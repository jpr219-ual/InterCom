#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

'''Feedback suppression with frequency domain subtraction.'''

import logging
import numpy as np
import minimal
import buffer
        
class Feedback_Suppression(buffer.Buffering):
    def __init__(self):
        super().__init__()
        logging.info(__doc__)
        
        # Parámetros mejorados para la supresión de retroalimentación
        self.suppression_factor = 0.1  # Factor de supresión (más conservador)
        self.adaptation_rate = 0.01    # Tasa de adaptación del filtro
        self.noise_floor = 0.01        # Piso de ruido para evitar división por cero
        
        # Buffer para el historial de salida (para estimar el feedback)
        self.output_history = np.zeros((self.chunks_to_buffer * 2, 
                                      minimal.args.frames_per_chunk, 
                                      minimal.args.number_of_channels), 
                                     dtype=np.float32)
        
        # Estimación espectral del feedback
        self.feedback_estimate = None
        self.feedback_spectrum = None
        
        # Estado inicial
        self.history_index = 0
        self.is_initialized = False

    def update_feedback_estimate(self, output_signal):
        """Actualiza la estimación del feedback basado en la salida reciente."""
        # Almacenar la salida en el historial
        self.output_history[self.history_index] = output_signal.astype(np.float32)
        self.history_index = (self.history_index + 1) % len(self.output_history)
        
        if not self.is_initialized:
            # Inicializar la estimación en la primera iteración
            self.feedback_estimate = output_signal.astype(np.float32).copy()
            self.is_initialized = True
        else:
            # Actualizar la estimación de forma adaptativa
            alpha = self.adaptation_rate
            self.feedback_estimate = (1 - alpha) * self.feedback_estimate + alpha * output_signal.astype(np.float32)

    def spectral_subtraction(self, input_signal, feedback_estimate):
        """
        Aplica sustracción espectral para suprimir el feedback.
        
        Parameters
        ----------
        input_signal : np.ndarray
            Señal de entrada del micrófono
        feedback_estimate : np.ndarray
            Estimación del feedback
            
        Returns
        -------
        np.ndarray
            Señal con feedback suprimido
        """
        output_signal = np.zeros_like(input_signal, dtype=np.int16)
        
        for channel in range(minimal.args.number_of_channels):
            # Convertir a float para procesamiento
            input_ch = input_signal[:, channel].astype(np.float32) / 32768.0
            feedback_ch = feedback_estimate[:, channel].astype(np.float32) / 32768.0
            
            # Aplicar ventana de Hann para reducir artefactos
            window = np.hanning(len(input_ch))
            input_windowed = input_ch * window
            feedback_windowed = feedback_ch * window
            
            # Transformada de Fourier
            input_fft = np.fft.rfft(input_windowed)
            feedback_fft = np.fft.rfft(feedback_windowed)
            
            # Magnitudes espectrales
            input_magnitude = np.abs(input_fft)
            feedback_magnitude = np.abs(feedback_fft)
            input_phase = np.angle(input_fft)
            
            # Sustracción espectral con protección contra división por cero
            # Usamos un enfoque de filtrado espectral en lugar de sustracción directa
            suppression_mask = np.ones_like(input_magnitude)
            
            # Aplicar supresión donde el feedback es significativo
            feedback_threshold = 0.1  # Umbral para considerar feedback significativo
            feedback_regions = feedback_magnitude > feedback_threshold
            
            # Crear máscara de supresión
            suppression_mask[feedback_regions] = np.maximum(
                1.0 - self.suppression_factor * (feedback_magnitude[feedback_regions] / 
                      (input_magnitude[feedback_regions] + self.noise_floor)),
                0.1  # Mínimo de supresión para evitar cancelación completa
            )
            
            # Aplicar la máscara
            output_magnitude = input_magnitude * suppression_mask
            
            # Reconstruir la señal
            output_fft = output_magnitude * np.exp(1j * input_phase)
            output_ch = np.fft.irfft(output_fft)
            
            # Asegurar la misma longitud
            if len(output_ch) > len(input_ch):
                output_ch = output_ch[:len(input_ch)]
            elif len(output_ch) < len(input_ch):
                output_ch = np.pad(output_ch, (0, len(input_ch) - len(output_ch)))
            
            # Remover ventana y convertir de vuelta a int16
            output_ch = output_ch / window  # Compensar la ventana
            output_ch = np.clip(output_ch * 32768.0, -32768, 32767).astype(np.int16)
            
            output_signal[:, channel] = output_ch
        
        return output_signal

    def nlp_processing(self, input_signal, feedback_estimate):
        """
        Non-Linear Processing (NLP) - método alternativo más agresivo.
        Útil cuando la sustracción espectral no es suficiente.
        """
        output_signal = np.zeros_like(input_signal, dtype=np.int16)
        
        for channel in range(minimal.args.number_of_channels):
            input_ch = input_signal[:, channel].astype(np.float32)
            feedback_ch = feedback_estimate[:, channel].astype(np.float32)
            
            # Detección de correlación (feedback)
            correlation = np.correlate(input_ch, feedback_ch, mode='same')
            correlation_threshold = 0.7 * np.max(np.abs(correlation))
            
            # Aplicar gate no-lineal en regiones con alta correlación
            output_ch = input_ch.copy()
            high_correlation = np.abs(correlation) > correlation_threshold
            
            # Atenuar fuertemente las regiones con feedback
            output_ch[high_correlation] = output_ch[high_correlation] * 0.1
            
            output_signal[:, channel] = np.clip(output_ch, -32768, 32767).astype(np.int16)
        
        return output_signal

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS

        # 1. Actualizar la estimación del feedback con la salida anterior
        if hasattr(self, 'previous_output'):
            self.update_feedback_estimate(self.previous_output)

        # 2. Aplicar supresión de feedback a la entrada actual
        if self.is_initialized:
            ADC_processed = self.spectral_subtraction(ADC, self.feedback_estimate)
            
            # Si todavía hay problemas, aplicar NLP adicional
            if self.detect_feedback(ADC_processed):
                ADC_processed = self.nlp_processing(ADC_processed, self.feedback_estimate)
        else:
            ADC_processed = ADC

        # 3. Procesamiento normal del buffer
        packed_chunk = self.pack(self.chunk_number, ADC_processed)
        self.send(packed_chunk)
        chunk = self.unbuffer_next_chunk()
        self.play_chunk(DAC, chunk)

        # 4. Guardar la salida actual para la próxima iteración
        self.previous_output = DAC.copy()

        return ADC_processed

    def detect_feedback(self, signal, threshold=0.9):
        """
        Detecta la presencia de feedback analizando la señal.
        
        Parameters
        ----------
        signal : np.ndarray
            Señal a analizar
        threshold : float
            Umbral para detección de feedback
            
        Returns
        -------
        bool
            True si se detecta feedback
        """
        # Calcular la relación de pico a RMS
        rms = np.sqrt(np.mean(signal.astype(np.float32)**2))
        peak = np.max(np.abs(signal.astype(np.float32)))
        
        if rms > 0:
            crest_factor = peak / rms
            return crest_factor > threshold
        return False

    def adaptive_suppression(self, input_signal, feedback_level):
        """
        Ajusta adaptativamente el factor de supresión basado en el nivel de feedback.
        """
        # Feedback level debería ser entre 0 y 1
        min_suppression = 0.1
        max_suppression = 0.8
        
        adaptive_factor = min_suppression + (max_suppression - min_suppression) * feedback_level
        self.suppression_factor = adaptive_factor
        
        return self.spectral_subtraction(input_signal, self.feedback_estimate)


class Feedback_Suppression__verbose(Feedback_Suppression, buffer.Buffering__verbose):
    def __init__(self):
        super().__init__()
        self.feedback_suppression_stats = {
            'signals_processed': 0,
            'feedback_detected_count': 0,
            'average_suppression_db': 0
        }
    
    def spectral_subtraction(self, input_signal, feedback_estimate):
        """Versión verbose que registra estadísticas."""
        output = super().spectral_subtraction(input_signal, feedback_estimate)
        
        # Calcular estadísticas
        input_power = np.mean(input_signal.astype(np.float32)**2)
        output_power = np.mean(output.astype(np.float32)**2)
        
        if input_power > 0 and output_power > 0:
            suppression_db = 10 * np.log10(input_power / output_power)
            self.feedback_suppression_stats['average_suppression_db'] = (
                self.feedback_suppression_stats['average_suppression_db'] * 
                self.feedback_suppression_stats['signals_processed'] + suppression_db
            ) / (self.feedback_suppression_stats['signals_processed'] + 1)
        
        self.feedback_suppression_stats['signals_processed'] += 1
        
        return output
    
    def detect_feedback(self, signal, threshold=0.9):
        detected = super().detect_feedback(signal, threshold)
        if detected:
            self.feedback_suppression_stats['feedback_detected_count'] += 1
        return detected
    
    def print_final_averages(self):
        super().print_final_averages()
        if self.feedback_suppression_stats['signals_processed'] > 0:
            print(f"\nFeedback Suppression Statistics:")
            print(f"Signals processed: {self.feedback_suppression_stats['signals_processed']}")
            print(f"Feedback detected: {self.feedback_suppression_stats['feedback_detected_count']} times")
            print(f"Average suppression: {self.feedback_suppression_stats['average_suppression_db']:.2f} dB")

try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")

if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        logging.warning("argcomplete not working :-/")
    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.show_stats or minimal.args.show_samples or minimal.args.show_spectrum:
        intercom = Feedback_Suppression__verbose()
    else:
        intercom = Feedback_Suppression()
        
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()