using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Random = UnityEngine.Random;

public class CartPoleVoluntaryAgent : Agent
{
    [SerializeField] private Rigidbody2D cartRigidbody;
    [SerializeField] private Rigidbody2D poleRigidbody;
    [SerializeField] private Vector2 initialPoleAngleOffsetMinMax;
    [SerializeField] private float maxAngle;
    [SerializeField] private float rewardPerSecond;
    [SerializeField] private float maxEpisodeSeconds;
    [SerializeField] private float impulsePerSecond;
    [SerializeField] private float maxNextActionDelaySeconds;

    private int movementDirection;
    private float currentEpisodeTimeSeconds;
    private float decidedNextActionDelay;
    private float timeSinceLastDecision;
    private Vector3 poleContactPointResetLocalPosition;

    protected override void Awake()
    {
        poleContactPointResetLocalPosition = poleRigidbody.transform.localPosition;
    }

    public override void OnEpisodeBegin()
    {
        cartRigidbody.transform.localPosition = Vector3.zero;
        
        cartRigidbody.velocity = Vector2.zero;
        cartRigidbody.angularVelocity = 0.0f;

        var poleTransform = poleRigidbody.transform;
        poleTransform.localPosition = poleContactPointResetLocalPosition;
        poleTransform.localRotation = Quaternion.Euler(0, 0, Random.Range(initialPoleAngleOffsetMinMax.x, initialPoleAngleOffsetMinMax.y));
        
        poleRigidbody.velocity = Vector2.zero;
        poleRigidbody.angularVelocity = 0.0f;
        
        currentEpisodeTimeSeconds = 0f;

        timeSinceLastDecision = float.PositiveInfinity;
        decidedNextActionDelay = 0;
        movementDirection = 0;
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(cartRigidbody.transform.localPosition.x / 4.5f);
        sensor.AddObservation(cartRigidbody.velocity.x / 6f);
        
        sensor.AddObservation(poleRigidbody.transform.localRotation.eulerAngles.z < 180 ? 
            poleRigidbody.transform.localRotation.eulerAngles.z / 45f : 
            (poleRigidbody.transform.localRotation.eulerAngles.z - 360f) / 45f);
        
        sensor.AddObservation(poleRigidbody.velocity / 6f);
        sensor.AddObservation(poleRigidbody.angularVelocity / 100f);
    }
    
    public virtual void FixedUpdate()
    {
        if (poleRigidbody.transform.localRotation.eulerAngles.z >= maxAngle && poleRigidbody.transform.localRotation.eulerAngles.z <= 360 - maxAngle)
        {
            EndEpisode();
            return;
        }

        currentEpisodeTimeSeconds += Time.fixedDeltaTime;
        
        if (currentEpisodeTimeSeconds >= maxEpisodeSeconds)
        {
            EndEpisode();
            return;
        }
        
        timeSinceLastDecision += Time.fixedDeltaTime;
        
        if (timeSinceLastDecision >= decidedNextActionDelay)
        {
            timeSinceLastDecision = 0;
            RequestDecision();
        }
        
        cartRigidbody.AddForce(Vector2.right * (movementDirection * impulsePerSecond * Time.fixedDeltaTime), ForceMode2D.Impulse);
        AddReward(rewardPerSecond * Time.fixedDeltaTime);
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActs = actionsOut.DiscreteActions;
        
        if (Input.GetKey(KeyCode.A))
        {
            discreteActs[0] = 0;
        } 
        else if (Input.GetKey(KeyCode.D))
        {
            discreteActs[0] = 2;
        }
        else
        {
            discreteActs[0] = 1;
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        movementDirection = actionBuffers.DiscreteActions[0] - 1;

        float delay = actionBuffers.ContinuousActions[0];
        decidedNextActionDelay = delay * maxNextActionDelaySeconds;
    }
}