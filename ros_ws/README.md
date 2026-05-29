# ROS 1 Noetic workspace — Sprint 2 bridgeable baseline

Workspace catkin para la migración del proyecto a ROS 1 Noetic + Gazebo
Classic. Estado actual: **baseline puenteable**.

Ya no estamos en puro scaffolding:

- el contenedor `tfg-ros` compila y arranca
- existe un labeler de habitaciones por centroides/Voronoi
- existe un bridge file-backed para `/map`, `/odom` y `tf`
- el task manager ya resuelve instrucciones simples a goals 2D
- el contrato Python `SemanticGoal -> NavigationBackend` ya puede hablar
  con `rosbridge` sin importar `rospy`

Lo que **todavía no** está conectado es el backend final de navegación
física/simulada (`move_base` con Gazebo/HSR y verificación visual real).
Eso se cableará sobre esta base, sin tocar el pipeline Habitat/HSSD.

## Arranque rápido

```bash
# 1) Construir la imagen del contenedor ROS (~6 GB; primera vez ~8 min)
cd docker && docker compose build tfg-ros

# 2) Levantar el contenedor
docker compose up -d tfg-ros

# 3) Entrar
docker exec -it tfg-ros bash

# Dentro del contenedor:
# 4) Construir el workspace
cd /ros_ws && catkin build

# 5) Smoke test (rosbridge + 4 stubs + RViz)
roslaunch vlmap_bringup minimal.launch
```

El contenedor queda configurado para cargar ROS Noetic y el overlay de
`/ros_ws/devel` automaticamente en `bash`, `bash -c` y `bash -lc`, asi que no
hace falta ejecutar `source /opt/ros/noetic/setup.bash` ni
`source /ros_ws/devel/setup.bash` a mano.

Con labeler cargado:

```bash
roslaunch vlmap_bringup minimal.launch \
  use_rviz:=false \
  occupancy_path:=/tmp/occ.json \
  room_centroids_path:=/tmp/centroids.json
```

En otra terminal del contenedor:

```bash
# Verificar que los 4 stubs están vivos
rostopic echo /vlmap/heartbeat
```

Deberías ver mensajes alternados de `vlmap_semantic_server`,
`vlmap_task_manager`, `habitat_ros_bridge` y `yoloe_verifier`.

## Estructura del workspace

```
ros_ws/src/
├── vlmap_msgs/             # SemanticGoal.msg + QueryRoom.srv + ResolveRoom.srv
├── vlmap_semantic_server/  # responde queries semánticas (rooms, candidatos)
├── vlmap_task_manager/     # NL → SemanticGoal / move_base_simple/goal
├── habitat_ros_bridge/     # publica /map, /odom, /tf desde archivos o bridge vivo
├── yoloe_verifier/         # verificación visual con YOLOE
└── vlmap_bringup/          # launch + rviz + costmap params
```

## Labeler de habitaciones por centroides

Sprint 2 ya tiene implementado el primer bloque reusable en:

`ros_ws/src/vlmap_semantic_server/src/vlmap_semantic_server/centroid_voronoi.py`

Qué hace:

- carga un conjunto de centroides semánticos (uno por habitación)
- construye una partición de Voronoi sobre el grid de ocupación
- asigna solo celdas navegables
- deja obstáculos y unknown sin etiqueta
- expone una clase `VoronoiRoomProvider` con la misma interfaz conceptual
  que la capa room-aware actual (`get_room_at_cell`, `get_room_centroid`,
  `list_rooms`)

Formato JSON de centroides:

```json
{
  "rooms": [
    {"label": "kitchen", "row": 120, "col": 340},
    {"label": "bedroom", "row": 410, "col": 220},
    {"label": "office", "row": 500, "col": 140}
  ]
}
```

Notas de diseño:

- las coordenadas están expresadas en celdas del `OccupancyGrid`
- el desempate entre centroides equidistantes es determinista: gana el
  primero del fichero
- este módulo no depende de ROS ni de Habitat, así que puede testearse en
  host normal con `pytest`

Tests offline:

```bash
cd /workspace
pytest tests/test_ros_voronoi_room_labeler.py -v
```

Parámetros ya soportados por `vlmap_semantic_server_node`:

- `~occupancy_path`
  - ruta a un `.npy` o `.json` con el grid de ocupación 2D
- `~room_centroids_path`
  - ruta al JSON de centroides mostrado arriba

Si ambos parámetros están presentes, el nodo construye `VoronoiRoomProvider`
al arrancar y publica el número de habitaciones cargadas en
`/vlmap/heartbeat`.

Servicio ya activo:

- `/vlmap/query_room` (`vlmap_msgs/QueryRoom`)
  - entrada: categoría libre, por ejemplo `laptop`
  - salida: habitaciones ordenadas + score por habitación
  - implementación actual:
    - usa `vlmaps.utils.room_priors.compute_room_priors` si el submódulo está
      disponible en `/workspace/third_party/vlmaps`
    - cae a distribución uniforme si el helper no puede importarse
  - en esta fase no consume todavía evidencia del heatmap ni estado del
    episodio; es un primer servicio semántico usable sobre el nuevo labeler

- `/vlmap/resolve_room` (`vlmap_msgs/ResolveRoom`)
  - entrada: nombre semántico de habitación
  - salida: habitación exacta + punto 2D en `/map`
  - usa el centroide Voronoi y `map_resolution` / `map_origin_*`

## Bridge actual

`habitat_ros_bridge_node` ya soporta un modo **file-backed**:

- carga un grid 2D desde `occupancy_path` (`.npy` o `.json`)
- publica `/map` (`nav_msgs/OccupancyGrid`)
- publica `/odom`
- publica `tf: map -> odom -> base_link`
- opcionalmente lee una pose viva desde `pose_path` (`{"x","y","yaw"}`)

Esto no acopla todavía ROS al runtime de Habitat; deja una interfaz estable que
después podrá alimentarse desde:

- un publisher vivo en `tfg-sim`
- Gazebo
- HSR físico

Además, el servidor semántico ya puede consumir un contexto vivo por topic:

- `/vlmap/room_context` (`std_msgs/String`, JSON)
  - contiene habitaciones, poses 2D y, opcionalmente, priors por categoría
  - si está presente, `/vlmap/query_room` y `/vlmap/resolve_room` usan ese
    contexto dinámico antes de caer al modo Voronoi/file-backed

Herramienta auxiliar para publicar ese contexto desde un JSON:

```bash
rosrun vlmap_semantic_server publish_room_context.py \
  _context_path:=/tmp/room_context.json
```

## Task manager actual

`vlmap_task_manager_node` ya soporta:

- suscripción a `/vlmap/instruction` (`std_msgs/String`)
- parsing ligero de instrucciones tipo:
  - `find laptop`
  - `search for mug`
  - `find the laptop in the office`
- consulta a:
  - `/vlmap/query_room`
  - `/vlmap/resolve_room`
- publicación de:
  - `/vlmap/semantic_goal` (`vlmap_msgs/SemanticGoal`)
  - `/move_base_simple/goal` (`geometry_msgs/PoseStamped`)

Tubería actual:

`instruction -> room ranking -> room centroid -> SemanticGoal -> ROS nav goal`

## Result bridge y smoke test end-to-end

La línea ROS ya cierra un ciclo completo de prueba:

`SemanticGoal -> /move_base_simple/goal -> /move_base/result -> /vlmap/navigation_result`

Piezas nuevas:

- `vlmap_navigation_result_node`
  - escucha `/vlmap/semantic_goal`
  - recuerda el `navigation_token` enviado desde `tfg-sim`
  - escucha resultados terminales de `/move_base/result`
  - publica un JSON backend-neutral en `/vlmap/navigation_result`
- `mock_move_base_node`
  - solo para smoke tests
  - escucha `/move_base_simple/goal`
  - publica un `MoveBaseActionResult` exitoso tras un retardo pequeño

Esto permite validar hoy mismo:

- publicación de goal semántico
- publicación de goal 2D
- retorno de `NavigationResult`
- preservación de `room_id` y `object_class`

Smoke test útil:

```bash
roslaunch vlmap_bringup minimal.launch \
  use_rviz:=false \
  use_mock_move_base:=true \
  occupancy_path:=/tmp/occ.json \
  room_centroids_path:=/tmp/centroids.json
```

Y desde Python:

```python
from tfg_nav_contracts import GoalType, RosNavigationBackend, SemanticGoal

backend = RosNavigationBackend(host="127.0.0.1", port=9090)
token = backend.submit_goal(
    SemanticGoal(
        type=GoalType.ROOM,
        map_pose=(1.75, 0.75),
        room_id="office",
        object_class="laptop",
        metadata={"source": "smoke"},
    )
)
result = backend.wait_result(token, timeout_s=5.0)
```

CLI equivalente para probar el contrato desde `tfg-sim` o desde el host:

```bash
# Desde tfg-sim, usando el nombre Docker del contenedor ROS.
python tools/send_ros_semantic_goal.py \
  --host tfg-ros \
  --x 0.50 \
  --y 0.00 \
  --yaw 0.00 \
  --room office \
  --object laptop \
  --timeout 30

# Desde el host, usando el puerto publicado por docker-compose.
python tools/send_ros_semantic_goal.py \
  --host 127.0.0.1 \
  --x 0.50 \
  --y 0.00 \
  --yaw 0.00 \
  --room office \
  --object laptop \
  --timeout 30
```

Prerequisitos de esa prueba:

- `tfg-ros` debe tener activo `roslaunch vlmap_bringup hsr_gazebo_move_base.launch`
- debe existir un solo `rosbridge_websocket` escuchando en `9090`
- `tfg-sim` debe estar reconstruido o tener instalado `roslibpy`, incluido ya en `docker/requirements.txt`
- el goal está en coordenadas métricas del frame ROS `/map`, no en celdas Habitat

La ruta de datos queda:

```text
tfg-sim / host
  tools/send_ros_semantic_goal.py
    -> rosbridge ws://tfg-ros:9090
      -> /vlmap/semantic_goal       (vlmap_msgs/SemanticGoal)
      -> /move_base_simple/goal     (geometry_msgs/PoseStamped)
        -> move_base
          -> /move_base/result
            -> vlmap_navigation_result_bridge
              -> /vlmap/navigation_result (std_msgs/String JSON)
```

Resultado esperado:

- `success=True`
- `found=True`
- `actual_room="office"`

## Contrato de verificación visual

La parte ROS ya expone una interfaz estable para delegar la verificación visual
fuera de Noetic:

- servicio `/vlmap/verify_object` (`vlmap_msgs/VerifyObject`)
- topic `/vlmap/verification/request` (`std_msgs/String`, JSON)
- topic `/vlmap/verification/result` (`std_msgs/String`, JSON)

Flujo:

1. un cliente ROS pide verificación con `object_class` y, opcionalmente,
   topics/campos específicos
2. `yoloe_verifier_node` publica un JSON en
   `/vlmap/verification/request`
3. un worker externo (por ejemplo `tfg-sim`) responde por
   `/vlmap/verification/result`
4. el servicio devuelve `success/found/message/metadata`

Mientras el worker real no exista, se puede usar el mock:

```bash
rosrun yoloe_verifier mock_verify_worker.py
```

Esto deja preparado el siguiente reemplazo:

- hoy: `mock_move_base_node`
- mañana: `move_base` real en Gazebo
- después: backend de navegación del HSR sim
- por último: HSR físico

## Gazebo + move_base real

El primer launch con un robot Gazebo ya está disponible:

```bash
roslaunch vlmap_bringup gazebo_move_base.launch gui:=false
```

Incluye:

- Gazebo Classic con `empty.world`
- robot diferencial mínimo `simple_diffbot.urdf`
- laser Gazebo publicando `/scan`
- diff-drive publicando `/odom` y consumiendo `/cmd_vel`
- `move_base` real
- `vlmap_navigation_result_bridge`
- `rosbridge`
- mapa demo file-backed servido como `/map` y `/static_map`

Validación realizada:

- `catkin build` dentro de `tfg-ros`: OK
- `/odom`: publicado por Gazebo
- `/scan`: publicado por Gazebo
- goal enviado desde `RosNavigationBackend` a `(0.5, 0.0)`
- odometría avanzó hasta aproximadamente `x=0.27`
- `wait_result()` devolvió:
  - `success=True`
  - `found=True`
  - `actual_room="origin"`
  - `message="Goal reached."`

Limitación actual:

- el robot es un stand-in mínimo, no HSR
- el mapa demo estático no representa una casa real
- la siguiente fase es sustituir `simple_diffbot` por HSR-sim o un robot
  compatible y alimentar el mapa desde una escena realista

## Contrato Python desacoplado para dos contenedores

La separación real entre `tfg-sim` y `tfg-ros` vive en:

`src/tfg_nav_contracts/`

Piezas clave:

- `SemanticGoal`
  - contrato Python mínimo, dependency-free
  - espejo de `vlmap_msgs/SemanticGoal`
- `NavigationBackend`
  - interfaz común para ejecución Habitat o ROS
- `RosNavigationBackend`
  - ya implementado
  - usa `roslibpy` y `rosbridge` en `ws://tfg-ros:9090`
  - publica:
    - `/vlmap/semantic_goal`
    - `/move_base_simple/goal`
    - `/move_base/cancel`
  - acepta resultados por:
    - `/move_base/result`
    - o un topic JSON intermedio durante la migración

Decisión importante de diseño:

- el backend ROS asume **un único goal activo cada vez**
- eso encaja con la política estratégica actual, que ya ejecuta búsqueda
  secuencial objetivo a objetivo
- si más adelante hace falta paralelismo, se amplía el contrato, pero no
  se mezcla ahora con el cierre del TFG

Ventaja:

- `tfg-sim` conserva Habitat/HSSD, VLMaps y evaluación
- `tfg-ros` conserva ROS, Gazebo, `move_base`, `tf` y drivers
- la comunicación se limita a un contrato estable y a `rosbridge`
- no hace falta instalar ROS dentro del entorno conda del stack semántico

## Arquitectura objetivo

Dos contenedores con responsabilidades estrictamente separadas:

| contenedor | qué corre |
|---|---|
| `tfg-sim` (ya existe) | Habitat-Sim, CLIP, YOLOE, VLMaps, razonamiento por habitaciones (Python 3.11 conda) |
| `tfg-ros` (este) | ROS 1 Noetic, Gazebo Classic, `move_base`, `rviz`, rosbridge (Python 3.8 sistema) |

Comunicación: **rosbridge sobre WebSocket** en el puerto 9090. Esto
evita instalar ROS dentro del env conda y evita instalar
torch/CLIP/YOLOE en el env ROS. Inspirado en MoMa-LLM
(Honerkamp et al., RSS SemRob 2024).

## Separación de stacks

Esta migración está diseñada para **no tocar ni degradar** el pipeline
Habitat/HSSD existente.

La separación es esta:

| stack | vive en | responsabilidad |
|---|---|---|
| Habitat/HSSD + VLMaps + YOLOE | `tfg-sim` | evaluación HSSD, construcción VLMap, razonamiento actual |
| ROS1 Noetic + move_base + bringup | `tfg-ros` | navegación ROS, topics, servicios, bridge a Gazebo/HSR |

Regla operativa:

- si una pieza depende de `third_party/vlmaps`, HSSD o el harness actual,
  se mantiene del lado `tfg-sim`
- si una pieza define interfaces ROS, mapas, pose, goals o servicios,
  se implementa del lado `tfg-ros`

Eso permite mantener HSSD como banco de validación permanente mientras se
construye en paralelo el stack ROS/Gazebo/HSR.

## Roadmap (resumen del PDF de migración)

- **Sprint 0** (este) — Contratos: `SemanticGoal` + `NavigationBackend`.
  Ver `src/tfg_nav_contracts/`.
- **Sprint 1** (este) — Workspace ROS, stubs, contenedor, smoke test.
- **Sprint 2** — Recolector RGB+depth+pose (en `habitat_ros_bridge`),
  labeler de habitaciones por centroides + Voronoi (en `vlmap_semantic_server`),
  goals semánticos publicados a `move_base` (en `vlmap_task_manager`).
- **Sprint 3** — Phase G + objetos pequeños (`yoloe_verifier`),
  re-cableado del harness 2×2 contra el entrypoint ROS.
- **Sprint 4** — Validación cross-stack (Habitat vs ROS) y, si hay
  HSR físico disponible, smoke test sobre el robot real.

## Dónde encontrar el código que se reutilizará en Sprint 2

| stub ROS | módulos Python existentes a envolver |
|---|---|
| `vlmap_semantic_server` | `third_party/vlmaps/vlmaps/map/vlmap.py`, `vlmaps/utils/room_provider.py`, `vlmaps/utils/room_priors.py` |
| `vlmap_task_manager` | `vlmaps/policy/strategic_policy.py`, `vlmaps/policy/executor.py`, `application/interactive_object_nav.compute_heatmap` |
| `habitat_ros_bridge` | `application/interactive_object_nav.py` (sensor observations, pose) |
| `yoloe_verifier` | `vlmaps/utils/yoloe_utils.py` |

Cada módulo stub lleva además un comentario `# TODO Sprint 2:` con la
lista exacta de imports.

## Variables de entorno relevantes

| variable | uso |
|---|---|
| `ROS_MODE` | `shell` (default), `core` o `launch:<file>` |
| `ROS_MASTER_URI` | preconfigurado a `http://tfg-ros:11311` |
| `DISPLAY` | reenviado del host para RViz / Gazebo |

## Restricciones de esta línea ROS

- El contenedor `tfg-sim` no se ha tocado.
- El submódulo `third_party/vlmaps/` no se ha tocado.
- Los nodos ROS no consumen torch ni CLIP ni Habitat.
- El contrato Python no importa `rospy`.
- Ningún workflow del proyecto principal (build VLMap, evaluación 2×2,
  fase heatmap/orquestador) cambia.
