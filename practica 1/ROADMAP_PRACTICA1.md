# Ruta de Trabajo: Práctica 1 - Control Avanzado de Turtlesim (ROS 2)

Este documento sirve como guía paso a paso y lista de control para la realización de la Práctica 1 de Robótica (ROS 2).

---

## 🎯 Objetivos de la Práctica

Desarrollar una aplicación en ROS 2 que extienda el simulador `turtlesim` con las siguientes características:

1. **Movimiento continuo y combinado**:
   - `W`: Avanzar
   - `S`: Retroceder
   - `A`: Girar a la izquierda
   - `D`: Girar a la derecha
   - *Capacidad de avanzar y girar simultáneamente* (curvas fluidas al mantener p. ej. `W` + `A` o `W` + `D`).

2. **Acciones mediante Servicios**:
   - `SPACE`: Habilitar / deshabilitar el trazo de dibujo (servicio `/turtle1/set_pen`).
   - `C`: Limpiar todo el dibujo de la pantalla (servicio `/clear`).
   - `R`: Reiniciar la posición de la tortuga al centro de la pantalla y mirando hacia arriba (`x=5.54, y=5.54, theta=π/2 = 1.5708` con el servicio `/turtle1/teleport_absolute`).

3. **Parámetros configurables dinámicamente**:
   - Parámetro para seleccionar el nivel de logging (`DEBUG`, `INFO`, `WARN`, etc.).
   - Parámetro para modificar la velocidad de la tortuga en tiempo de ejecución (`ros2 param set`).

4. **Lanzador automático (Launch File)**:
   - Archivo `.launch.py` que inicie `turtlesim_node` y configure los parámetros de la práctica.

---

## 🗺️ Roadmap de Ejecución

- [x] **Fase 0: Preparación del Entorno**
  - [x] Verificar instalación de ROS 2 (Jazzy) y paquete `turtlesim` en contenedor Docker.
  - [x] Comprobar librerías auxiliares (`pynput`, `x11-apps`).
  - [x] Configurar el entorno con montaje sincronizado de carpetas.

- [x] **Fase 1: Análisis de Interfaces de Turtlesim**
  - [x] Topic de velocidad: `/turtle1/cmd_vel` (`geometry_msgs/msg/Twist`).
  - [x] Servicios: `/turtle1/set_pen`, `/clear`, `/turtle1/teleport_absolute`.
  - [x] Pruebas interactivas y verificación de la ventana gráfica en Docker.

- [x] **Fase 2: Desarrollo del Nodo de Teleoperación**
  - [x] Crear el paquete ROS 2 (`turtle_control`).
  - [x] Implementar captura de pulsación simultánea de teclas (`pynput.keyboard.Listener`).
  - [x] Implementar temporizador periódico para publicar en `/turtle1/cmd_vel`.

- [x] **Fase 3: Integración de Servicios y Parámetros**
  - [x] Añadir clientes de servicio asíncronos para `SPACE` (set_pen toggle), `C` (clear) y `R` (teleport_absolute al centro mirando hacia arriba).
  - [x] Declarar y gestionar parámetros de velocidad (`speed`, `angular_speed`).
  - [x] Declarar y gestionar parámetro de nivel de log (`log_level`).

- [x] **Fase 4: Archivo Launch y Empaquetado**
  - [x] Configurar `setup.py` con los ejecutables y carpeta `launch`.
  - [x] Crear el fichero `turtlesim_teleop.launch.py` que inicializa Turtlesim con los parámetros adecuados.
  - [x] Compilar el paquete con `colcon build`.

- [x] **Fase 5: Validación y Pruebas Finales**
  - [x] Lanzar el launch file conjunto (`ros2 launch turtle_control turtlesim_teleop.launch.py`).
  - [x] Prueba de avance + giro simultáneo (teclas W + A o W + D en curvas).
  - [x] Prueba de las teclas de servicios (`SPACE`, `C`, `R`).
  - [x] Cambio dinámico de velocidad en caliente mediante `ros2 param set /teleop_turtle speed <nuevo_valor>`.
  - [x] Generación de la Memoria técnica en PDF (`Memoria_Practica1.pdf`).
  - [x] Empaquetado limpio del Workspace / src en archivo ZIP para entrega (`entrega_practica1.zip` y `entrega_src.zip`).
